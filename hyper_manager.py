#!/usr/bin/env python3
# Copyright 2019 Brad Martin.  All rights reserved.

import collections
import datetime
import functools
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time

from matplotlib import pyplot
import numpy
from PySide2 import QtCore, QtGui, QtWidgets
from tensorboard.backend.event_processing.event_accumulator \
    import EventAccumulator

import project_paths

_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
_CMDLINE = [os.path.join(_ROOT_DIR, 'training.py'),
            '--dataset', os.path.join(_ROOT_DIR, 'dataset.npz'),
            '--intervals', '6']

_M_H = 60.  # minutes per hour
_S_M = 60.  # seconds per minute
_S_H = _S_M * _M_H  # seconds per hour


def _parse_logs(logs_dir):
    """Reads a tensorboard log directory.

    Returns an Nx5 numpy array, where N is the number of logged events in the
    log, and the 5 columns are:
     * The amount of training time when the event was logged,
     * The training loss,
     * The validation loss,
     * The training error (excluding regularization penalties),
     * The validation error (excluding regularization penalties).
    """
    rel_time = 0.
    full_log = None
    try:
        for subdir in sorted(os.listdir(logs_dir)):
            acc = EventAccumulator(os.path.join(logs_dir, subdir))
            acc.Reload()

            try:
                train_loss = acc.Scalars('epoch_loss')
                val_loss = acc.Scalars('epoch_val_loss')
            except KeyError:
                continue
            if not train_loss or not val_loss:
                continue

            try:
                train_error = acc.Scalars('epoch_masked_mse')
                val_error = acc.Scalars('epoch_val_masked_mse')
            except KeyError:
                # Some files might not log this separately.  In that case
                # assume the loss is the error metric.
                train_error, val_error = train_loss, val_loss

            this_log = numpy.array([
                (tl.wall_time - train_loss[0].wall_time + rel_time,
                 tl.value, vl.value, te.value, ve.value)
                for tl, vl, te, ve in zip(
                        train_loss, val_loss, train_error, val_error)])
            if full_log is None:
                full_log = this_log
            else:
                full_log = numpy.concatenate((full_log, this_log))
            rel_time += train_loss[-1].wall_time - train_loss[0].wall_time
    except FileNotFoundError:
        pass
    return full_log


def _get_variance(log_data):
    """Calculate variance.

    Returns the variance for each row of data provided with columns as per the
    result of _parse_logs().  Variance is defined here as 1 - training_error /
    validation_error.
    """
    return 1. - log_data[:, 3] / log_data[:, 4]


def _filter_log(full_log):
    """Applies a moving-average filter on the given log data.

    Given input in the same form as the output of _parse_logs, returns output
    in the same form, where all columns have had a moving-average filter
    applied.
    """
    FILTER_M = 10.
    if full_log is None:
        return None
    total_m = (full_log[-1, 0] - full_log[0, 0]) / _S_M
    samples_per_m = full_log.shape[0] / total_m
    filter_width = int(round(FILTER_M * samples_per_m))
    if full_log.shape[0] < filter_width:
        return numpy.array([])
    filtered = numpy.stack(
        [numpy.convolve(
            full_log[:, i], numpy.ones(filter_width) / filter_width,
            mode='valid')
         for i in range(full_log.shape[1])],
        axis=1)
    return filtered


def _get_stats(full_log, filtered=None):
    """Gets log error stats.

    Given the results of _parse_logs() and _filter_log() respectively, returns
    a tuple of four elements:
     * Minimum validation error in the log.
     * Validation error at the end of the log.
     * Variance (defined as 1 - training_error / validation_error) at the end
       of the log.
     * The rate at which the validation error was changing towards the end of
       the log, per minute of training time.
     * The intercept of the linear fit used to estimate the error rate.

    Returns a tuple of Nones if there is no input log data.

    If the filtered output is not provided, it is implicitly computed from
    full_log.

    """
    if (full_log is not None) and (filtered is None):
        filtered = _filter_log(full_log)
    if (full_log is None) or (filtered.shape[0] == 0):
        return None, None, None, None, None

    min_error = numpy.amin(full_log[:, 4])
    cur_error = filtered[-1, 4]
    cur_variance = _get_variance(filtered[-1:, :])[0]
    # Do a linear fit to the latter half of the log data.
    start_idx = full_log.shape[0] // 2
    data = full_log[start_idx:, :]
    A = numpy.hstack([data[:, 0:1], numpy.ones((data.shape[0], 1))])
    B = data[:, 4:5]
    p, _, _, _ = numpy.linalg.lstsq(A, B, rcond=None)
    error_rate = p[0][0] * 60.  # Per minute not per second.
    error_intercept = p[1][0]

    return (min_error,
            cur_error, cur_variance,
            error_rate, error_intercept)


class ManagerState(QtCore.QObject):
    _HYPERSET_FILE = 'hyperset.json'
    _SESSION_FILE = 'session.json'
    _LOG_FILE = 'training.log'
    _OVERWRITTEN_LINE_RE = re.compile('[^\r\n]*\r')
    _DEFAULT_DATA = {
        'args': '',
        'disabled': False,
        'this_run_s': None,
        'total_training_s': 0.,
        'min_test_error': None,
        'cur_test_error': None,
        'test_error_rate': None,
        'cur_variance': None,
    }

    class Priority:
        COUNT = 4
        NONE, TIME, RATE, TTZ = range(COUNT)
        PRETTY = {
            # No prioritization: equal weighting.
            NONE: 'None',
            # Prioritize sets that have less training time over those that
            # have more.
            TIME: 'Training Time',
            # Prioritize sets where the error is estimated to be decreasing
            # faster.
            RATE: 'Error Rate',
            # Prioritize sets that would be expected to reach zero error
            # sooner, if the current error rate estimate held indefinitely.
            # Obviously this won't happen, but this does give an effective
            # blend between prioritizing error rate, and prioritizing sets
            # that already have a relatively low error.
            TTZ: '"Time to Zero"'}

    # The key of a previously-existing hyperset that was updated is passed
    # through the signal.  If None, the entire table should be treated as
    # changed.
    set_updated_signal = QtCore.Signal(str)
    # Indicates that session properties other than set data have changed.
    session_updated_signal = QtCore.Signal()

    def __init__(self):
        super(ManagerState, self).__init__()
        self._session_path = None
        self._hypersets = {}
        self._timer_id = None
        self._running_set_key = None
        self._subprocess = None
        self._subprocess_log = None
        self._last_time = None
        self._console_text = ''
        self._forced_run_key = None
        self._priority = None
        self._variance_threshold = None

        self._reset()

    def _reset(self):
        self._session_path = None
        self._hypersets = {}
        self._forced_run_key = None
        self._priority = self.Priority.TTZ
        self._variance_threshold = None

    @property
    def session_path(self):
        return self._session_path

    @property
    def hypersets(self):
        return self._hypersets

    @property
    def running(self):
        return self._timer_id is not None

    @property
    def console_text(self):
        return self._console_text

    @property
    def priority(self):
        return self._priority

    @property
    def variance_threshold(self):
        return self._variance_threshold

    def set_session_path(self, path):
        assert os.path.exists(path)
        if self.running:
            self.stop_session()
        self._reset()

        self._session_path = path
        self._try_load_session_props()
        for setfile in glob.glob(os.path.join(path, '*', self._HYPERSET_FILE)):
            self._load_hyperset(setfile)
        self.set_updated_signal.emit(None)
        self.session_updated_signal.emit()

    def add_hyperset(self, set_args):
        # Create a lookup key for each hyperparam set based on hashing the
        # training args string.
        hasher = hashlib.sha256()
        hasher.update(set_args.encode('utf8'))
        key = hasher.hexdigest()
        if key in self._hypersets:
            raise ValueError(
                'A hyperparam set with arguments %r already exists!' %
                set_args)
        self._hypersets[key] = dict(self._DEFAULT_DATA)
        self._hypersets[key]['args'] = set_args
        os.mkdir(os.path.join(self._session_path, key))
        self._write_set_data(key)
        # key didn't previously exist, so reset everything.
        self.set_updated_signal.emit(None)

    def disable_hyperset(self, key, disabled):
        self._hypersets[key]['disabled'] = disabled
        self._write_set_data(key)
        self.set_updated_signal.emit(key)
        if disabled and (self._running_set_key == key):
            self.stop_session()
            self.run_session()

    def run_session(self):
        assert not self.running
        self._timer_id = self.startTimer(1000, QtCore.Qt.VeryCoarseTimer)
        self.session_updated_signal.emit()

    def stop_session(self):
        assert self.running
        self.killTimer(self._timer_id)
        self._timer_id = None
        if self._subprocess is not None:
            self._subprocess.terminate()
            try:
                self._subprocess.wait(20)
            except subprocess.TimeoutExpired:
                print('Process ignored TERM, trying KILL')
                self._subprocess.kill()
                self._subprocess.wait(20)
            assert self._update_subprocess(ignore_errors=True)
        assert self._running_set_key is None
        self._console_text = 'Not running'
        self.session_updated_signal.emit()

    def get_history(self, key):
        return _parse_logs(
            os.path.join(self._session_path, key, project_paths.LOGS_DIR))

    def refresh_stats(self, key):
        self._update_stats(key)
        self._write_set_data(key)
        self.set_updated_signal.emit(key)

    def set_priority(self, val):
        assert val in range(self.Priority.COUNT)
        self._priority = val
        self._save_session_props()
        self.session_updated_signal.emit()

    def set_variance_threshold(self, val):
        if val is not None:
            assert (val > 0.) and (val <= 1.)
        self._variance_threshold = val
        self._save_session_props()
        self.session_updated_signal.emit()

    def run_hyperset(self, key):
        assert key in self._hypersets
        if self.running:
            self.stop_session()
        self._forced_run_key = key
        self.run_session()

    # QObject override
    def timerEvent(self, _):
        if not self.running:
            # Got a stale event.
            return

        if self._subprocess is not None:
            self._update_subprocess()

        # Find a session to run if we don't already have one.
        if self._running_set_key is None:
            self._running_set_key = self._choose_runnable_set()
            if self._running_set_key is None:
                print('No runnable sets')
                self.stop_session()
                return
            self._hypersets[self._running_set_key]['this_run_s'] = 0.
            assert self._subprocess is None
            set_wd = os.path.join(self._session_path, self._running_set_key)
            checkpoint = self._find_existing_checkpoint(set_wd)
            cmd = _CMDLINE + (
                ['--resume-from', checkpoint]
                if checkpoint is not None
                else self._hypersets[self._running_set_key]['args'].split(' '))
            self._subprocess_log = open(os.path.join(set_wd, self._LOG_FILE),
                                        'at', encoding='utf8')
            self._subprocess_log.write(
                '*** %s RUN at %s (%r)\n' %
                ('RESUME' if checkpoint is not None else 'START',
                 time.ctime(),
                 "' '".join(cmd)))
            self._subprocess_log.flush()
            self._subprocess = subprocess.Popen(
                cmd, stdout=self._subprocess_log, stderr=subprocess.STDOUT,
                cwd=set_wd, encoding='utf8', universal_newlines=True)
            self._last_time = time.monotonic()

    def _try_load_session_props(self):
        props_fn = os.path.join(self._session_path, self._SESSION_FILE)
        if not os.path.exists(props_fn):
            return
        with open(props_fn, 'rt', encoding='utf8') as f:
            data = json.load(f)
        self._priority = data.get('priority', self._priority)
        self._variance_threshold = data.get('variance_threshold',
                                            self._variance_threshold)
        assert self._priority in range(self.Priority.COUNT)
        if self._variance_threshold is not None:
            assert ((self._variance_threshold > 0.) and
                    (self._variance_threshold <= 1.))

    def _save_session_props(self):
        props_fn = os.path.join(self._session_path, self._SESSION_FILE)
        with open(props_fn + '.tmp', 'wt', encoding='utf8') as f:
            json.dump({'priority': self._priority,
                       'variance_threshold': self._variance_threshold},
                      f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(props_fn + '.tmp', props_fn)

    def _load_hyperset(self, setfile):
        with open(setfile, 'rt', encoding='utf8') as f:
            data = json.load(f)
        key = os.path.basename(os.path.dirname(setfile))
        # Even if the data was saved while running, we shouldn't indicate that
        # run status now.
        data['this_run_s'] = None
        # Start with the default data and then selectively update it with the
        # loaded state, to ensure that we have exactly the fields we expect in
        # the state going forward, even if the stored file used an old schema.
        self._hypersets[key] = dict(self._DEFAULT_DATA)
        for k in self._hypersets[key].keys():
            if k in data:
                self._hypersets[key][k] = data[k]

    def _write_set_data(self, key):
        set_fn = os.path.join(self._session_path, key, self._HYPERSET_FILE)
        with open(set_fn + '.tmp', 'wt', encoding='utf8') as f:
            json.dump(self._hypersets[key], f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(set_fn + '.tmp', set_fn)

    def _update_subprocess(self, ignore_errors=False):
        assert self._subprocess is not None
        running_set_key = self._running_set_key
        assert running_set_key is not None
        running_set = self._hypersets[running_set_key]

        result = self._subprocess.poll()
        with open(
                os.path.join(
                    self._session_path, running_set_key, self._LOG_FILE),
                'rb') as f:
            try:
                f.seek(-25000, 2)
            except OSError:
                f.seek(0)
            # Real console output has carriage returns that overwrite lines,
            # but these don't show up in Qt very well.  We'll just detect any
            # lines that end with a carriage return, and discard them
            # entirely.
            self._console_text = self._OVERWRITTEN_LINE_RE.sub(
                '', f.read().decode('utf8'))

        now = time.monotonic()
        dt = now - self._last_time
        self._last_time = now
        running_set['this_run_s'] += dt
        running_set['total_training_s'] += dt

        if result is not None:
            # This can end up being re-entrant if we're not careful:
            # refreshing stats causing the running set to exceed the variance
            # threshold, disabling it, so we stop the running set, which
            # involves updating its stats which again tries to stop it
            # running.  Best to indicate as soon as possible that we're not
            # running anymore to prevent this.
            self._subprocess = None
            self._running_set_key = None

            if not ignore_errors and (result != 0):
                print('Error %r with args %r (key %r):' %
                      (result, running_set['args'],
                       running_set_key))
                print(self._console_text)
                running_set['disabled'] = True
            self._subprocess_log.write('*** END RUN at %s\n' % time.ctime())
            self._subprocess_log.close()
            running_set['this_run_s'] = None
            self._update_stats(running_set_key)
            self._last_time = None

        self._write_set_data(running_set_key)
        self.set_updated_signal.emit(running_set_key)
        return result is not None

    def _choose_runnable_set(self):
        if self._forced_run_key is not None:
            k = self._forced_run_key
            self._forced_run_key = None
            assert k in self._hypersets
            return k

        runnable_sets = [k for k, v in self._hypersets.items()
                         if not v['disabled']]
        if not runnable_sets:
            return None
        weights = numpy.array([self._get_priority_weight(k)
                               for k in runnable_sets])
        cum_weights = numpy.cumsum(weights)
        rval = random.uniform(0., cum_weights[-1])
        return runnable_sets[numpy.argmax(cum_weights >= rval)]

    def _get_priority_weight(self, key):
        data = self._hypersets[key]
        if any(v is None for f, v in data.items() if f != 'this_run_s'):
            # Prioritize at least one interval on any new sets so we can
            # establish some stats on which to base priority going forward.
            return 10.

        if self._priority == self.Priority.NONE:
            weight = 1.
        elif self._priority == self.Priority.TIME:
            weight = 1. / data['total_training_s']
        elif self._priority == self.Priority.RATE:
            weight = -data['test_error_rate']
        elif self._priority == self.Priority.TTZ:
            weight = -data['test_error_rate'] / data['min_test_error']
        else:
            assert False, 'Unknown priority %r' % self._priority

        EPSILON = 1e-6  # Cap small/negative weights at this positive value.
        return max(weight, EPSILON)

    def _find_existing_checkpoint(self, set_dir):
        checkpoints = glob.glob(os.path.join(
            set_dir, project_paths.CHECKPOINT_DIR, '*.json'))
        if not checkpoints:
            return None
        checkpoints.sort()
        return os.path.relpath(checkpoints[-1], start=set_dir)

    def _update_stats(self, key):
        # NOTE: It is necessary to rewrite the key state and issue an update
        # signal following this method.
        assert key in self._hypersets
        # (Discard the intercepts.)
        full_log = self.get_history(key)
        filtered = _filter_log(full_log)
        if (full_log is not None) and \
           (filtered.shape[0] > 0) and \
           (self._variance_threshold is not None):
            variance = _get_variance(filtered)
            mask = variance > self._variance_threshold
            filtered_idx = numpy.argmax(mask)
            if mask[filtered_idx]:
                # First disable the set.
                self.disable_hyperset(key, True)
                # Truncate full_log by the detected amount.  We'll leave the
                # filtered log alone, since the 'current' stats should not be
                # thresholded.
                #
                # Remember that the two arrays don't have the same number of
                # rows, so we need to match by time instead.
                exceeded_t = filtered[filtered_idx][0]
                full_idx = numpy.argmax(full_log[:, 0] >= exceeded_t)
                if full_idx:
                    full_log = full_log[:full_idx]

        for k, v in zip(['min_test_error', 'cur_test_error', 'cur_variance',
                         'test_error_rate'],
                        _get_stats(full_log, filtered=filtered)):
            self._hypersets[key][k] = v


class ManagerStateTableAdapter(QtCore.QAbstractTableModel):
    _HEADERS = ['Args', 'This Run', 'Total Training',
                'Min Test Error', 'Cur Test Error', 'Error Rate',
                'Cur Variance', 'TTZ', 'Key']
    TTZ_COL = _HEADERS.index('TTZ')
    KEY_COL = _HEADERS.index('Key')

    def __init__(self, parent, state):
        super(ManagerStateTableAdapter, self).__init__(parent)
        self._parent = parent
        self._state = state
        state.set_updated_signal.connect(self._handle_state_changed)
        self._data = []
        self._disabled = []
        self._sort_column = 0
        self._sort_order = QtCore.Qt.DescendingOrder
        self._sorted_keys = []

        self._handle_state_changed(None)

    # QAbstractTableModel interface implementation

    def rowCount(self, _=QtCore.QModelIndex()):
        ret = len(self._data)
        return ret

    def columnCount(self, _=QtCore.QModelIndex()):
        ret = len(self._HEADERS)
        return ret

    def headerData(self, section, orientation, role):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            return self._HEADERS[section]
        return str(section)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        column = index.column()
        row = index.row()

        if role == QtCore.Qt.DisplayRole:
            return self._data[row][column]

        color_group = (
            QtGui.QPalette.ColorGroup.Disabled if self._disabled[row]
            else QtGui.QPalette.ColorGroup.Normal)

        if role == QtCore.Qt.BackgroundRole:
            return self._parent.palette().brush(
                color_group, QtGui.QPalette.ColorRole.Base)

        if role == QtCore.Qt.ForegroundRole:
            return self._parent.palette().brush(
                color_group, QtGui.QPalette.ColorRole.Text)

        return None

    def sort(self, column, order=QtCore.Qt.AscendingOrder):
        self._sort_column = column
        self._sort_order = order
        self._handle_state_changed(None)

    # Other methods
    def get_key(self, row_index):
        return self._data[row_index][self.KEY_COL]

    def _handle_state_changed(self, key):
        def fmt_float(f):
            return '%.4g' % f if f is not None else ''

        def row_data(k, d):
            return [
                d['args'],
                (str(datetime.timedelta(seconds=round(d['this_run_s'])))
                 if d['this_run_s'] is not None else ''),
                str(datetime.timedelta(seconds=round(d['total_training_s']))),
                fmt_float(d['min_test_error']),
                fmt_float(d['cur_test_error']),
                fmt_float(d['test_error_rate']),
                fmt_float(d['cur_variance']),
                '' if ((d['cur_test_error'] is None) or
                       (d['test_error_rate'] is None))
                else '--' if d['test_error_rate'] >= 0.
                else str(int(d['cur_test_error'] / -d['test_error_rate'])),
                k,
            ]

        sorted_keys = self._get_sorted_keys()
        if (key is None) or \
           (key not in self._state.hypersets) or \
           (sorted_keys != self._sorted_keys):
            self.beginResetModel()
            self._sorted_keys = sorted_keys
            self._data = [row_data(k, self._state.hypersets[k])
                          for k in self._sorted_keys]
            self._disabled = [self._state.hypersets[k]['disabled']
                              for k in self._sorted_keys]
            self.endResetModel()
        else:
            idx = sorted_keys.index(key)
            self._data[idx] = row_data(key, self._state.hypersets[key])
            self._disabled[idx] = self._state.hypersets[key]['disabled']
            self.dataChanged.emit(self.index(idx, 0),
                                  self.index(idx, self.columnCount() - 1))

    def _get_sorted_keys(self):
        def key_on_field(f, k):
            ret = self._state.hypersets[k][f]
            return ret if ret is not None else float('-inf')

        def key_on_ttz(k):
            error = self._state.hypersets[k]['cur_test_error']
            rate = self._state.hypersets[k]['test_error_rate']
            if (error is None) or (rate is None):
                return float('-inf')
            if rate >= 0.:
                return float('inf')
            return error / -rate

        key = ([functools.partial(key_on_field, f) for f in
                ['args', 'this_run_s', 'total_training_s', 'min_test_error',
                 'cur_test_error', 'test_error_rate', 'cur_variance']] +
               [key_on_ttz, None])[self._sort_column]
        return sorted(self._state.hypersets.keys(), key=key,
                      reverse=(self._sort_order == QtCore.Qt.DescendingOrder))


class HypersetTable(QtWidgets.QTableView):
    selection_updated_signal = QtCore.Signal()

    def __init__(self, parent, state):
        super(HypersetTable, self).__init__(parent)
        self._state = state
        self._model = ManagerStateTableAdapter(self, state)
        self._model.modelAboutToBeReset.connect(self._handle_model_reset)

        def add_action(label, handler):
            action = QtWidgets.QAction(label)
            action.triggered.connect(handler)
            self.addAction(action)
            return action

        self._disable_set_action = add_action(
            'Disable set(s) from running', self._handle_disable_set)
        self._disable_set_action.setCheckable(True)
        self._plot_error_action = add_action(
            'Plot validation error', self._handle_plot_error)
        self._reset_stats_action = add_action(
            'Refresh error statistics', self._handle_refresh_stats)
        self._run_set_action = add_action(
            'Run this set', self._handle_run_set)

        self._selected_keys = []

        # Calls on self seem to start invoking slots which may look for data
        # that doesn't exist yet if they don't come after all variable
        # initialization in the constructor.
        self.setModel(self._model)
        hz = self.horizontalHeader()
        vt = self.verticalHeader()
        hz.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        vt.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        hz.setStretchLastSection(True)
        self.setSortingEnabled(True)
        self.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.setMinimumHeight(8 * self.fontMetrics().lineSpacing())
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.sortByColumn(ManagerStateTableAdapter.TTZ_COL,
                          QtCore.Qt.AscendingOrder)

    @property
    def selected_keys(self):
        return self._selected_keys

    def selectionChanged(self, selected, deselected):
        super(HypersetTable, self).selectionChanged(selected, deselected)
        self._selected_keys = [
            self._model.get_key(row)
            for row in set(index.row() for index in self.selectedIndexes())]
        self._disable_set_action.setChecked(
            self._state.hypersets[self._selected_keys[0]]['disabled']
            if self._selected_keys else False)
        self._run_set_action.setEnabled(len(self._selected_keys) == 1)
        self.selection_updated_signal.emit()

    def _handle_model_reset(self):
        self.clearSelection()
        self._selected_keys = []

    def _handle_disable_set(self, checked):
        for key in self._selected_keys:
            self._state.disable_hyperset(key, checked)
        self.clearSelection()
        self._selected_keys = []

    def _handle_plot_error(self):
        COLORS = 'cmykrgb'

        keys = self._selected_keys or self._state.hypersets.keys()
        all_data = collections.OrderedDict()
        for k in keys:
            d = self._state.get_history(k)
            if d is not None:
                all_data[k] = (d, _filter_log(d))
        if not all_data:
            return

        pyplot.figure()
        pyplot.subplot(2, 1, 1)
        for i, (key, (data, filtered)) in enumerate(all_data.items()):
            _, _, _, rate, intercept = _get_stats(data, filtered=filtered)
            color = COLORS[i % len(COLORS)]
            pyplot.plot(data[:, 0] / _S_H, data[:, 4], color=color,
                        marker='.', linestyle='', markersize=1)
            pyplot.plot(filtered[:, 0] / _S_H, filtered[:, 4], color=color,
                        label=self._state.hypersets[key]['args'])
            pyplot.plot(data[:, 0] / _S_H,
                        intercept + rate * data[:, 0] / _M_H,
                        color=color, linestyle='--')
        pyplot.title('Error')
        pyplot.xlabel('Time (h)')
        pyplot.yscale('log')
        pyplot.grid()
        pyplot.legend()

        pyplot.subplot(2, 1, 2)
        for i, (key, (data, filtered)) in enumerate(all_data.items()):
            color = COLORS[i % len(COLORS)]
            pyplot.plot(data[:, 0] / _S_H, _get_variance(data), color=color,
                        marker='.', linestyle='', markersize=1)
            pyplot.plot(filtered[:, 0] / _S_H, _get_variance(filtered),
                        color=color, label=self._state.hypersets[key]['args'])
        pyplot.title('Variance')
        pyplot.xlabel('Time (h)')
        pyplot.ylim(-0.1, 1.1)
        pyplot.grid()
        pyplot.legend()

        pyplot.show()

    def _handle_refresh_stats(self):
        for key in (self._selected_keys or self._state.hypersets.keys()):
            self._state.refresh_stats(key)

    def _handle_run_set(self):
        assert len(self._selected_keys) == 1
        self._state.run_hyperset(self._selected_keys[0])


class MainWidget(QtWidgets.QSplitter):
    def __init__(self, parent, state):
        super(MainWidget, self).__init__(QtCore.Qt.Vertical, parent)
        self.setChildrenCollapsible(False)

        self.table = HypersetTable(self, state)
        self.addWidget(self.table)

        font = QtGui.QFont('Courier', 10)
        self.text = QtWidgets.QTextEdit(self)
        self.text.setReadOnly(True)
        self.text.setEnabled(False)
        self.text.setCurrentFont(font)
        self.text.setMinimumHeight(4 * self.text.fontMetrics().lineSpacing())
        self.text.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.addWidget(self.text)

        self.setStretchFactor(0, 30)
        self.setStretchFactor(1, 1)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, initial_session):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Hyper Manager')

        self._state = ManagerState()
        self._state.set_updated_signal.connect(self._handle_set_updated)
        self._state.session_updated_signal.connect(
            self._handle_session_updated)

        self._widget = MainWidget(self, self._state)
        self._widget.table.selection_updated_signal.connect(
            self._handle_selection_updated)
        self.setCentralWidget(self._widget)

        self._menu = self.menuBar()

        self._file_menu = self._menu.addMenu('File')
        self._file_menu.addAction('New session...', self._handle_new_session)
        self._file_menu.addAction('Load session...', self._handle_load_session)
        self._file_menu.addAction('Exit', self._exit_app)

        self._sets_menu = self._menu.addMenu('Sets')
        self._sets_menu.addAction(
            'Add hyperparam set...', self._handle_add_set)
        self._variance_threshold_action = self._sets_menu.addAction(
            'Variance threshold...', self._handle_variance_threshold)
        self._sets_menu.addSeparator()
        self._set_actions = []
        for table_action in self._widget.table.actions():
            action = QtWidgets.QAction(table_action.text())
            action.setCheckable(table_action.isCheckable())
            action.triggered[bool].connect(table_action.triggered[bool])
            self._set_actions.append((action, table_action))
            self._sets_menu.addAction(action)

        self._run_menu = self._menu.addMenu('Run')
        self._run_action = self._run_menu.addAction(
            'Run session', self._handle_run_session)
        self._stop_action = self._run_menu.addAction(
            'Stop session', self._handle_stop_session)
        self._run_menu.addSeparator()
        pri_menu = self._run_menu.addMenu('Prioritize by')
        self._priority_actions = [
            pri_menu.addAction(ManagerState.Priority.PRETTY[p],
                               functools.partial(self._handle_priority, p))
            for p in range(ManagerState.Priority.COUNT)]
        group = QtWidgets.QActionGroup(self)
        group.setExclusive(True)
        for a in self._priority_actions:
            a.setActionGroup(group)
            a.setCheckable(True)

        self._status = self.statusBar()

        if initial_session is not None:
            self._state.set_session_path(initial_session)
        else:
            self._handle_session_updated()

    # QMainWindow override
    def closeEvent(self, event):
        if self._state.running:
            self._state.stop_session()
        super(MainWindow, self).closeEvent(event)

    def _handle_session_updated(self):
        have_session = self._state.session_path is not None
        self._sets_menu.setEnabled(have_session)
        self._handle_selection_updated()
        self._run_menu.setEnabled(have_session)

        running = self._state.running
        self._run_action.setEnabled(not running)
        self._stop_action.setEnabled(running)
        for i, a in enumerate(self._priority_actions):
            a.setChecked(self._state.priority == i)

        self._widget.text.setEnabled(running)

        if self._state.session_path is not None:
            session = os.path.basename(self._state.session_path)
            if not session:
                session = os.path.dirname(self._state.session_path)
            if self._state.hypersets:
                status = 'Session %s: %s' % (
                    'running' if running else 'stopped', session)
            else:
                status = 'Empty session: %s' % session
        else:
            status = 'No session loaded'
        self._status.showMessage(status)

    def _handle_set_updated(self, _):
        self._widget.text.setPlainText(self._state.console_text)
        scroll = self._widget.text.verticalScrollBar()
        scroll.setValue(scroll.maximum())

    def _handle_selection_updated(self):
        have_selection = bool(self._widget.table.selected_keys)
        for my_action, table_action in self._set_actions:
            # This is a little hacky.  We want the 'disable' to be grayed out
            # with no selection, but every other action defaults to operating
            # on every entry if there is no selection, provided the table
            # action is enabled.
            if table_action.isCheckable():
                my_action.setEnabled(have_selection)
                my_action.setChecked(table_action.isChecked())
            else:
                my_action.setEnabled(table_action.isEnabled())

    def _handle_new_session(self):
        dialog = QtWidgets.QFileDialog(
            self, 'New session directory', project_paths.SESSIONS_DIR)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        if dialog.exec():
            session_dir = dialog.selectedFiles()[0]
            # NOTE: It seems as though the file dialog creates the directory
            # already?
            self._state.set_session_path(session_dir)

    def _handle_load_session(self):
        session_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Session directory', project_paths.SESSIONS_DIR,
            QtWidgets.QFileDialog.ShowDirsOnly)
        if session_dir:
            self._state.set_session_path(session_dir)

    def _exit_app(self):
        sys.exit(0)

    def _handle_add_set(self):
        set_args, ok = QtWidgets.QInputDialog.getText(
            self, 'New hyperparam set', 'Enter new hyperparam training args:')
        if ok:
            try:
                self._state.add_hyperset(set_args)
            except ValueError as e:
                QtWidgets.QMessageBox.warning(self, 'Set exists', str(e))

    def _handle_variance_threshold(self):
        val, ok = QtWidgets.QInputDialog.getDouble(
            self, 'Apply Variance Threshold',
            'Disables running hypersets when the \'current variance\' stat \n'
            'exceeds this value.  Also, the \'min error\' stat will apply \n'
            'to only that part of the log before this threshold is exceeded \n'
            '(requires refresh).  Set to zero to disable.',
            (self._state.variance_threshold
             if self._state.variance_threshold is not None else 0.),
            0., 1., 2, QtCore.Qt.WindowFlags(), 0.1)
        if ok:
            self._state.set_variance_threshold(val if val else None)

    def _handle_run_session(self):
        self._state.run_session()

    def _handle_stop_session(self):
        self._state.stop_session()

    def _handle_priority(self, priority):
        self._state.set_priority(priority)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    args = app.arguments()
    if (len(args) > 2) or ((len(args) == 2) and (args[1] == '--help')):
        print('Usage: %s [existing-session-name]' % args[0])
        sys.exit(1)
    window = MainWindow(args[1] if len(args) == 2 else None)
    window.show()
    sys.exit(app.exec_())
