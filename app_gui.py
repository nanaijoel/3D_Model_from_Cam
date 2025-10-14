import os
import subprocess
import glob
import threading

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QTextCursor
import open3d as o3d

from pipeline_runner import PipelineRunner


class Worker(QtCore.QThread):
    # Fortschritt/Logs/Ergebnis
    progress = QtCore.Signal(int, str)      # value, label
    log = QtCore.Signal(str)
    done = QtCore.Signal(str, str)          # ply_path, project_root
    failed = QtCore.Signal(str)

    # NEU: Anforderung aus Pipeline Schritt 10 -> GUI soll Frames abfragen
    # Übergabe: project_root, frame_basenames (Liste möglicher Frames)
    ask_manual_texture = QtCore.Signal(str, list)

    def __init__(self, base_dir: str, video_path: str, project_name: str, target_frames: int, manual_mode: bool):
        super().__init__()
        self.base_dir = base_dir
        self.video_path = video_path
        self.project_name = project_name
        self.target_frames = target_frames
        self.manual_mode = manual_mode

        # Sync-Objekte für blocking Callback
        self._manual_evt = None
        self._manual_result = None  # list[str] mit Basenames

    @QtCore.Slot(list)
    def receive_manual_texture_selection(self, names: list):
        # Wird vom GUI nach Dialog-OK/CANCEL aufgerufen
        if self._manual_evt is not None:
            self._manual_result = names or []
            self._manual_evt.set()

    def _ask_manual_texture_blocking(self, project_root: str, frame_basenames: list[str]) -> list[str]:
        """
        Diese Methode wird im Worker-Thread (aus der Pipeline) aufgerufen.
        Sie blockt bis das GUI (Main-Thread) die Auswahl geliefert hat.
        Rückgabe: Liste der ausgewählten Basenames.
        """
        self._manual_evt = threading.Event()
        self._manual_result = None
        # Signal an Main-Thread: Dialog öffnen
        self.ask_manual_texture.emit(project_root, frame_basenames)
        # Warten bis GUI antwortet
        self._manual_evt.wait()
        sel = self._manual_result or []
        # Aufräumen
        self._manual_evt = None
        self._manual_result = None
        return sel

    def run(self):
        try:
            # Logging/Progress Weiterleitung
            def on_log(msg: str):
                self.log.emit(msg)

            def on_progress(v, label):
                self.progress.emit(int(v), label)

            # ENV setzen: nur Modus, keine Views (die liefert der GUI-Dialog)
            env = os.environ
            if self.manual_mode:
                env["TEXTURE_MODE"] = "manual"
            else:
                env.pop("TEXTURE_MODE", None)
                env.pop("TEXTURE_VIEWS", None)

            runner = PipelineRunner(base_dir=self.base_dir, on_log=on_log, on_progress=on_progress)

            # Pipeline starten – mit Callback, das in Schritt 10 (Texturing) abgefragt wird
            # Signatur-Erweiterung: ask_manual_texture_cb ist optional
            try:
                ply_path, paths = runner.run(
                    self.video_path,
                    self.project_name,
                    self.target_frames,
                    ask_manual_texture_cb=self._ask_manual_texture_blocking
                )
            except TypeError:
                # Rückfall, falls deine PipelineRunner.run die neue Signatur noch nicht hat:
                ply_path, paths = runner.run(self.video_path, self.project_name, self.target_frames)

            self.done.emit(ply_path, paths.root)
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Model from Camera Video")
        self.resize(1000, 600)

        self.base_dir = os.path.abspath(".")
        self.video_path = ""
        self.last_ply = ""
        self.project_root = ""

        # Widgets
        self.btn_pick = QtWidgets.QPushButton("Choose video…")
        self.lbl_video = QtWidgets.QLabel("No file selected")
        self.lbl_video.setWordWrap(True)

        self.txt_project = QtWidgets.QLineEdit()
        self.txt_project.setPlaceholderText("Project name (e.g., buddha_better)")

        self.spin_frames = QtWidgets.QSpinBox()
        self.spin_frames.setRange(50, 200)
        self.spin_frames.setValue(120)
        self.spin_frames.setSuffix(" Frames")

        # NEU: Texturing-Modus (Auto / Manual)
        self.cmb_tex_mode = QtWidgets.QComboBox()
        self.cmb_tex_mode.addItems(["Auto (recommended)", "Manual (select frames at step 10)"])
        self.cmb_tex_mode.setCurrentIndex(0)

        self.btn_compute = QtWidgets.QPushButton("Compute")
        self.btn_compute.setEnabled(False)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.lbl_stage = QtWidgets.QLabel("…")
        self.chk_autoscroll = QtWidgets.QCheckBox("Autoscroll log")
        self.chk_autoscroll.setChecked(True)

        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)

        self.cmb_mesh = QtWidgets.QComboBox()
        self.cmb_mesh.setMinimumWidth(420)
        self.btn_choose_mesh = QtWidgets.QPushButton("Browse…")
        self.btn_show = QtWidgets.QPushButton("Open 3D Model")
        self.btn_show.setEnabled(False)

        self.btn_meshlab = QtWidgets.QPushButton("Open in MeshLab")
        self.btn_meshlab.setEnabled(False)

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow(self.btn_pick, self.lbl_video)
        form.addRow("Project name:", self.txt_project)
        form.addRow("Target frames:", self.spin_frames)
        form.addRow("Texturing:", self.cmb_tex_mode)
        form.addRow(self.btn_compute, self.progress)
        form.addRow("Stage:", self.lbl_stage)
        form.addRow(self.chk_autoscroll)

        mesh_row = QtWidgets.QHBoxLayout()
        mesh_row.addWidget(self.cmb_mesh, 1)
        mesh_row.addWidget(self.btn_choose_mesh)
        form.addRow("Select mesh:", mesh_row)

        top = QtWidgets.QWidget()
        top.setLayout(form)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(self.log)
        splitter.setStretchFactor(1, 1)

        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(bottom)
        bl.addWidget(self.btn_show)
        bl.addWidget(self.btn_meshlab)

        container = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(container)
        v.addWidget(splitter)
        v.addWidget(bottom)
        self.setCentralWidget(container)

        # Signals
        self.btn_pick.clicked.connect(self.pick_video)
        self.btn_compute.clicked.connect(self.compute)
        self.btn_show.clicked.connect(self.show_mesh)
        self.btn_choose_mesh.clicked.connect(self.choose_mesh)
        self.btn_meshlab.clicked.connect(self.open_in_meshlab)

        self.scan_meshes()
        self.worker = None

    # ---------- Utility / UI ----------

    def scan_meshes(self):
        self.cmb_mesh.blockSignals(True)
        self.cmb_mesh.clear()

        ply_paths = []
        for root, _, files in os.walk(self.base_dir):
            for fn in files:
                if fn.lower().endswith(".ply"):
                    ply_paths.append(os.path.join(root, fn))

        ply_paths.sort()
        for p in ply_paths:
            self.cmb_mesh.addItem(os.path.relpath(p, self.base_dir), userData=p)

        self.cmb_mesh.blockSignals(False)
        self.btn_show.setEnabled(self.cmb_mesh.count() > 0)
        self.btn_meshlab.setEnabled(self.cmb_mesh.count() > 0)

    def ensure_in_combo(self, path: str):
        if not path:
            return
        for i in range(self.cmb_mesh.count()):
            if self.cmb_mesh.itemData(i) == path:
                self.cmb_mesh.setCurrentIndex(i)
                self.btn_show.setEnabled(True)
                self.btn_meshlab.setEnabled(True)
                return
        try:
            base_dir_abs = os.path.abspath(self.base_dir)
            path_abs = os.path.abspath(path)
            common = os.path.commonpath([base_dir_abs, path_abs])
            label = os.path.relpath(path, self.base_dir) if common == base_dir_abs else os.path.basename(path)
        except Exception:
            label = os.path.basename(path)
        self.cmb_mesh.addItem(label, userData=path)
        self.cmb_mesh.setCurrentIndex(self.cmb_mesh.count() - 1)
        self.btn_show.setEnabled(True)
        self.btn_meshlab.setEnabled(True)

    def choose_mesh(self):
        filters = "PLY Mesh (*.ply);;All files (*)"
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose mesh", self.base_dir, filters)
        if not path:
            return
        self.ensure_in_combo(path)

    def pick_video(self):
        filters = ("Video files (*.mp4 *.MP4 *.mov *.MOV *.m4v *.M4V *.avi *.AVI *.mkv *.MKV *.webm *.WEBM *.wmv *.WMV);;"
                   "All files (*)")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose video", self.base_dir, filters)
        if not path:
            return
        self.video_path = path
        self.lbl_video.setText(path)
        self.btn_compute.setEnabled(True)
        self.progress.setValue(0)
        self.log.clear()

    def append_log(self, msg: str):
        self.log.append(msg)
        if self.chk_autoscroll.isChecked():
            self.log.moveCursor(QTextCursor.End)
            self.log.ensureCursorVisible()

    def set_progress(self, val: int, label: str):
        self.progress.setValue(val)
        self.lbl_stage.setText(label)

    # ---------- Main actions ----------

    def compute(self):
        if not self.video_path:
            QtWidgets.QMessageBox.warning(self, "Notice", "Please choose a video.")
            return
        project = self.txt_project.text().strip() or "default_project"
        target_frames = int(self.spin_frames.value())

        manual_mode = (self.cmb_tex_mode.currentIndex() == 1)

        self.btn_compute.setEnabled(False)
        self.worker = Worker(self.base_dir, self.video_path, project, target_frames, manual_mode)

        # Standard-Verbindungen
        self.worker.progress.connect(self.set_progress)
        self.worker.log.connect(self.append_log)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)

        # WICHTIG: Callback-Brücke (Pipeline -> GUI)
        self.worker.ask_manual_texture.connect(self.on_ask_manual_texture)

        self.worker.start()
        self.append_log("[ui] Pipeline started…")

    def on_done(self, ply_path: str, project_root: str):
        self.last_ply = ply_path
        self.project_root = project_root
        self.btn_compute.setEnabled(True)
        self.btn_show.setEnabled(True)
        self.btn_meshlab.setEnabled(True)
        self.append_log(f"[ui] Done: {ply_path}")
        self.scan_meshes()
        if os.path.isfile(ply_path):
            self.ensure_in_combo(ply_path)

    def on_failed(self, msg: str):
        self.btn_compute.setEnabled(True)
        self.append_log("[error]\n" + msg)
        QtWidgets.QMessageBox.critical(self, "Error", msg)

    # ---------- Mesh viewing ----------

    def _current_ply_path(self) -> str:
        ply_path = self.last_ply
        if self.cmb_mesh.currentIndex() >= 0:
            sel = self.cmb_mesh.currentData()
            if sel and os.path.isfile(sel):
                ply_path = sel
        return ply_path if (ply_path and os.path.isfile(ply_path)) else ""

    def open_in_meshlab(self):
        ply_path = self._current_ply_path()
        if not ply_path:
            QtWidgets.QMessageBox.information(self, "Notice", "No valid PLY file selected/found.")
            return

        appdir = os.path.expanduser("~/MeshLab/MeshLab2025.07-linux_x86_64")
        exe = os.path.join(appdir, "usr", "bin", "meshlab")

        env = os.environ.copy()
        # Qt-Umgebungen säubern
        for k in ["QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QT_STYLE_OVERRIDE",
                  "QT_API", "QT_QPA_PLATFORMTHEME", "PYTHONPATH", "LD_LIBRARY_PATH"]:
            env.pop(k, None)
        env["QT_QPA_PLATFORM"] = "xcb"

        subprocess.Popen([exe, os.path.abspath(ply_path)], cwd=appdir, env=env)
        self.append_log(f"[ui] Open in MeshLab: {ply_path}")

    def show_mesh(self):
        ply_path = self._current_ply_path()
        if not ply_path:
            QtWidgets.QMessageBox.information(self, "Notice", "No valid PLY file selected/found.")
            return
        try:
            pcd = o3d.io.read_point_cloud(ply_path)
            o3d.visualization.draw_geometries([pcd])
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error opening", str(e))

    # ---------- Manual-Dialog für Schritt 10 ----------

    @QtCore.Slot(str, list)
    def on_ask_manual_texture(self, project_root: str, frame_basenames: list):
        """
        Wird vom Worker (aus der Pipeline) aufgerufen, wenn TEXTURE_MODE=manual
        in Schritt 10 erreicht ist. Öffnet einen Dialog zur Auswahl und gibt
        die Basenames zurück an den Worker.
        """
        # Falls vom Pipeline-Callback keine Liste mitkam, lese selbst
        if not frame_basenames:
            frames_dir = os.path.join(project_root, "raw_frames")
            files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
            frame_basenames = [os.path.basename(f) for f in files]

        # Dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Select frames for texturing (Manual)")
        dlg.setModal(True)

        vbox = QtWidgets.QVBoxLayout(dlg)
        info = QtWidgets.QLabel(
            "Bitte wähle die Frames, die für das Texturing verwendet werden sollen.\n"
            "Hinweis: Mehrfachauswahl möglich (Strg/Shift)."
        )
        vbox.addWidget(info)

        listw = QtWidgets.QListWidget()
        listw.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        listw.addItems(frame_basenames)
        listw.setMinimumHeight(360)
        vbox.addWidget(listw)

        # Select All / None
        btn_row = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("Select All")
        btn_none = QtWidgets.QPushButton("Select None")
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        vbox.addLayout(btn_row)

        def do_all():
            listw.selectAll()

        def do_none():
            listw.clearSelection()

        btn_all.clicked.connect(do_all)
        btn_none.clicked.connect(do_none)

        # OK/Cancel
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        vbox.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        # Anzeigen & Ergebnis
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            names = [i.text() for i in listw.selectedItems()]
            self.append_log(f"[ui] Manual texture frames selected: {', '.join(names) if names else '(none)'}")
        else:
            names = []
            self.append_log("[ui] Manual texture selection canceled (fallback behavior applies)")

        # Ergebnis zurück an Worker (entblockt Pipeline)
        if self.worker is not None:
            self.worker.receive_manual_texture_selection(names)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    mw = MainWindow()
    mw.show()
    app.exec()
