#!/usr/bin/env python3
# Copyright 2019 Brad Martin.  All rights reserved.

import datetime
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time

import numpy
from PySide2 import QtCore, QtGui, QtWidgets
from tensorboard.backend.event_processing.event_accumulator \
    import EventAccumulator

import project_paths

_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
_CMDLINE = [os.path.join(_ROOT_DIR, 'training.py'),
            '--dataset', os.path.join(_ROOT_DIR, 'dataset.npz'),
            '--intervals', '6']


def _parse_logs(logs_dir):
    """Reads a tensorboard log dir and returns a tuple of three elements:
     * Minimum validation loss in the log.
     * Validation loss at the end of the log.
     * The rate at which the validation loss was changing towards the end of
       the log, per minute of training time.

    Returns three Nones if no logs are present at the given location.
    """
    rel_time = 0.
    full_log = None
    try:
        for subdir in sorted(os.listdir(logs_dir)):
            acc = EventAccumulator(os.path.join(logs_dir, subdir))
            acc.Reload()
            try:
                loss = acc.Scalars('epoch_val_loss')
            except KeyError:
                continue
            if not loss:
                continue
            this_log = numpy.array([
                (l.wall_time - loss[0].wall_time + rel_time, l.value)
                for l in loss])
            if full_log is None:
                full_log = this_log
            else:
                full_log = numpy.concatenate((full_log, this_log))
            rel_time += loss[-1].wall_time - loss[0].wall_time
    except FileNotFoundError:
        pass
    if full_log is None:
        return None, None, None

    min_loss = numpy.amin(full_log[:, 1])
    cur_loss = full_log[-1, 1]
    # Do a linear fit, but weight the later terms more aggressively than the
    # earlier ones.
    W = numpy.sqrt((numpy.arange(full_log.shape[0]) + 1)[:, numpy.newaxis])
    A = W * numpy.hstack([full_log[:, 0:1],
                          numpy.ones((full_log.shape[0], 1))])
    B = W * full_log[:, 1:2]
    p, _, _, _ = numpy.linalg.lstsq(A, B, rcond=None)
    loss_rate = p[0][0] * 60.  # Per minute not per second.

    return min_loss, cur_loss, loss_rate


class ManagerState(QtCore.QObject):
    _HYPERSET_FILE = 'hyperset.json'
    _LOG_FILE = 'training.log'
    _OVERWRITTEN_LINE_RE = re.compile('[^\r\n]*\r')

    updated_signal = QtCore.Signal()

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

    def _reset(self):
        if self.running:
            self.stop_session()
        self._session_path = None
        self._hypersets = {}

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

    def set_session_path(self, path):
        assert os.path.exists(path)
        self._reset()
        self._session_path = path
        for setfile in glob.glob(os.path.join(path, '*', self._HYPERSET_FILE)):
            self._load_hyperset(setfile)
        self.updated_signal.emit()

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
        data = {
            'args': set_args,
            'disabled': False,
            'this_run_s': None,
            'total_training_s': 0.,
            'min_test_loss': None,
            'cur_test_loss': None,
            'test_loss_rate': None,
        }
        self._hypersets[key] = data
        os.mkdir(os.path.join(self._session_path, key))
        self._write_set_data(key)
        self.updated_signal.emit()

    def disable_hyperset(self, key, disabled):
        self._hypersets[key]['disabled'] = disabled
        self._write_set_data(key)
        self.updated_signal.emit()

    def run_session(self):
        assert not self.running
        self._timer_id = self.startTimer(1000, QtCore.Qt.VeryCoarseTimer)

    def stop_session(self):
        assert self.running
        self.killTimer(self._timer_id)
        self._timer_id = None
        if self._subprocess is not None:
            self._subprocess.terminate()
            try:
                self._subprocess.wait(10)
            except subprocess.TimeoutExpired:
                self._subprocess.kill()
                self._subprocess.wait(10)
            assert self._update_subprocess(ignore_errors=True)
        assert self._running_set_key is None
        self._console_text = 'Not running'

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

    def _load_hyperset(self, setfile):
        with open(setfile, 'rt', encoding='utf8') as f:
            data = json.load(f)
        key = os.path.basename(os.path.dirname(setfile))
        # Even if the data was saved while running, we shouldn't indicate that
        # run status now.
        data['this_run_s'] = None
        self._hypersets[key] = data

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
                f.seek(-10000, 2)
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
            if not ignore_errors and (result != 0):
                print('Error %r with args %r (key %r):' %
                      (result, running_set['args'],
                       running_set_key))
                print(self._console_text)
                running_set['disabled'] = True
            self._subprocess_log.write('*** END RUN at %s\n' % time.ctime())
            self._subprocess_log.close()
            self._subprocess = None
            running_set['this_run_s'] = None
            for k, v in zip(
                    ['min_test_loss', 'cur_test_loss', 'test_loss_rate'],
                    _parse_logs(os.path.join(self._session_path,
                                             running_set_key,
                                             project_paths.LOGS_DIR))):
                running_set[k] = v
            self._running_set_key = None
            self._last_time = None

        self._write_set_data(running_set_key)
        self.updated_signal.emit()
        return result is not None

    def _choose_runnable_set(self):
        # TODO prioritize
        runnable_sets = [k for k, v in self._hypersets.items()
                         if not v['disabled']]
        if not runnable_sets:
            return None
        return random.choice(runnable_sets)

    def _find_existing_checkpoint(self, set_dir):
        checkpoints = glob.glob(os.path.join(
            set_dir, project_paths.CHECKPOINT_DIR, '*.json'))
        if not checkpoints:
            return None
        checkpoints.sort()
        return os.path.relpath(checkpoints[-1], start=set_dir)


class ManagerStateTableAdapter(QtCore.QAbstractTableModel):
    _HEADERS = ['Args', 'This Run', 'Total Training',
                'Min Test Loss', 'Cur Test Loss', 'Loss Rate', 'Key']
    _KEY_COL = _HEADERS.index('Key')

    def __init__(self, parent, state):
        super(ManagerStateTableAdapter, self).__init__(parent)
        self._parent = parent
        self._state = state
        state.updated_signal.connect(self._handle_state_changed)
        self._data = []
        self._disabled = []
        self._handle_state_changed()

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

    # Other methods
    def get_key(self, row_index):
        return self._data[row_index][self._KEY_COL]

    def _handle_state_changed(self):
        self.beginResetModel()

        def fmt_float(f):
            return '%.4g' % f if f is not None else ''

        def row_data(k, d):
            return [
                d['args'],
                (str(datetime.timedelta(seconds=round(d['this_run_s'])))
                 if d['this_run_s'] is not None else ''),
                str(datetime.timedelta(seconds=round(d['total_training_s']))),
                fmt_float(d['min_test_loss']),
                fmt_float(d['cur_test_loss']),
                fmt_float(d['test_loss_rate']),
                k,
            ]
        items = self._state.hypersets.items()
        # TODO sort
        self._data = [row_data(k, d) for k, d in items]
        self._disabled = [d['disabled'] for _, d in items]

        self.endResetModel()


class HypersetTable(QtWidgets.QTableView):
    selection_updated_signal = QtCore.Signal()

    def __init__(self, parent, state):
        super(HypersetTable, self).__init__(parent)
        self._state = state
        self._model = ManagerStateTableAdapter(self, state)
        self._model.modelAboutToBeReset.connect(self._handle_model_reset)

        hz = self.horizontalHeader()
        vt = self.verticalHeader()
        hz.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        vt.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        hz.setStretchLastSection(True)
        # TODO self.setSortingEnabled(True)
        self.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.setMinimumHeight(8 * self.fontMetrics().lineSpacing())
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)

        def add_action(label, handler):
            action = QtWidgets.QAction(label)
            action.triggered.connect(handler)
            self.addAction(action)
            return action

        self._queue_set_action = add_action(
            'Queue set(s) for running', self.handle_queue_set)
        self._disable_set_action = add_action(
            'Disable set(s) from running', self.handle_disable_set)
        self._disable_set_action.setCheckable(True)

        self._selected_keys = []

        # This call seems to start invoking slots which may look for data that
        # doesn't exist yet if it doesn't come last.
        self.setModel(self._model)

    @property
    def selected_keys(self):
        return self._selected_keys

    def selectionChanged(self, selected, deselected):
        super(HypersetTable, self).selectionChanged(selected, deselected)
        self._selected_keys = [self._model.get_key(row)
                               for row in set(index.row()
                                              for index in selected.indexes())]
        self._disable_set_action.setChecked(
            self._state.hypersets[self._selected_keys[0]]['disabled']
            if self._selected_keys else False)
        self.selection_updated_signal.emit()

    def handle_queue_set(self, _):
        raise NotImplementedError()

    def handle_disable_set(self, checked):
        for key in self.selected_keys:
            self._state.disable_hyperset(key, checked)

    def _handle_model_reset(self):
        self.clearSelection()
        self._selected_keys = None


class MainWidget(QtWidgets.QSplitter):
    def __init__(self, parent, state):
        super(MainWidget, self).__init__(QtCore.Qt.Vertical, parent)
        self.setChildrenCollapsible(False)

        self.table = HypersetTable(self, state)
        self.addWidget(self.table)

        self.text = QtWidgets.QTextEdit(self)
        self.text.setReadOnly(True)
        self.text.setEnabled(False)
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
        self._state.updated_signal.connect(self._handle_state_updated)

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

        self._status = self.statusBar()

        if initial_session is not None:
            self._state.set_session_path(initial_session)
        else:
            self._handle_state_updated()

    def _handle_state_updated(self):
        have_session = self._state.session_path is not None
        self._sets_menu.setEnabled(have_session)
        self._handle_selection_updated()

        self._run_menu.setEnabled(have_session)
        running = self._state.running
        self._run_action.setEnabled(not running)
        self._stop_action.setEnabled(running)

        self._widget.text.setEnabled(running)
        self._widget.text.setPlainText(self._state.console_text)
        self._widget.text.moveCursor(QtGui.QTextCursor.End)
        self._widget.text.ensureCursorVisible()

        if have_session:
            session = os.path.basename(self._state.session_path)
            if self._state.hypersets:
                status = 'Session %s: %s' % (
                    'running' if running else 'stopped', session)
            else:
                status = 'Empty session: %s' % session
        else:
            status = 'No session loaded'
        self._status.showMessage(status)

    def _handle_selection_updated(self):
        have_selection = bool(self._widget.table.selected_keys)
        for my_action, table_action in self._set_actions:
            my_action.setEnabled(have_selection)
            my_action.setChecked(table_action.isChecked())

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

    def _handle_run_session(self):
        self._state.run_session()

    def _handle_stop_session(self):
        self._state.stop_session()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    args = app.arguments()
    if (len(args) > 2) or ((len(args) == 2) and (args[1] == '--help')):
        print('Usage: %s [existing-session-name]' % args[0])
        sys.exit(1)
    window = MainWindow(args[1] if len(args) == 2 else None)
    window.show()
    sys.exit(app.exec_())
