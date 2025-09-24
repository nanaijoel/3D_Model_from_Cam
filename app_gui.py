import os
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QTextCursor

import open3d as o3d
from pipeline_runner import PipelineRunner


class Worker(QtCore.QThread):
    progress = QtCore.Signal(int, str)  # value, label
    log = QtCore.Signal(str)
    done = QtCore.Signal(str, str)      # ply_path, project_root
    failed = QtCore.Signal(str)

    def __init__(self, base_dir: str, video_path: str, project_name: str, target_frames: int):
        super().__init__()
        self.base_dir = base_dir
        self.video_path = video_path
        self.project_name = project_name
        self.target_frames = target_frames

    def run(self):
        try:
            def on_log(msg): self.log.emit(msg)
            def on_progress(v, label): self.progress.emit(int(v), label)
            runner = PipelineRunner(base_dir=self.base_dir, on_log=on_log, on_progress=on_progress)
            ply_path, paths = runner.run(self.video_path, self.project_name, self.target_frames)
            self.done.emit(ply_path, paths.root)
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Model from Camera Video")
        self.resize(980, 560)

        self.base_dir = os.path.abspath(".")
        self.video_path = ""
        self.last_ply = ""
        self.project_root = ""

        self.btn_pick = QtWidgets.QPushButton("Video auswählen…")
        self.lbl_video = QtWidgets.QLabel("Keine Datei gewählt")
        self.lbl_video.setWordWrap(True)

        self.txt_project = QtWidgets.QLineEdit()
        self.txt_project.setPlaceholderText("Projektname (z.B. buddha_better)")

        self.spin_frames = QtWidgets.QSpinBox()
        self.spin_frames.setRange(50, 2000)
        self.spin_frames.setValue(300)
        self.spin_frames.setSuffix(" Frames")

        self.btn_compute = QtWidgets.QPushButton("Compute")
        self.btn_compute.setEnabled(False)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.lbl_stage = QtWidgets.QLabel("…")
        self.chk_autoscroll = QtWidgets.QCheckBox("Autoscroll Log")
        self.chk_autoscroll.setChecked(True)

        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)

        self.cmb_mesh = QtWidgets.QComboBox()
        self.cmb_mesh.setMinimumWidth(420)
        self.btn_choose_mesh = QtWidgets.QPushButton("Auswählen…")
        self.btn_show = QtWidgets.QPushButton("Open 3D Model")
        self.btn_show.setEnabled(False)   # just activated, if at least one mesh available

        form = QtWidgets.QFormLayout()
        form.addRow(self.btn_pick, self.lbl_video)
        form.addRow("Projektname:", self.txt_project)
        form.addRow("Ziel-Frames:", self.spin_frames)
        form.addRow(self.btn_compute, self.progress)
        form.addRow("Schritt:", self.lbl_stage)
        form.addRow(self.chk_autoscroll)

        mesh_row = QtWidgets.QHBoxLayout()
        mesh_row.addWidget(self.cmb_mesh, 1)
        mesh_row.addWidget(self.btn_choose_mesh)
        form.addRow("Mesh auswählen:", mesh_row)

        top = QtWidgets.QWidget()
        top.setLayout(form)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(self.log)
        splitter.setStretchFactor(1, 1)

        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(bottom)
        bl.addWidget(self.btn_show)

        container = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(container)
        v.addWidget(splitter)
        v.addWidget(bottom)
        self.setCentralWidget(container)

        self.btn_pick.clicked.connect(self.pick_video)
        self.btn_compute.clicked.connect(self.compute)
        self.btn_show.clicked.connect(self.show_mesh)
        self.btn_choose_mesh.clicked.connect(self.choose_mesh)

        self.scan_meshes()

        self.worker = None

    def scan_meshes(self):
        self.cmb_mesh.blockSignals(True)
        self.cmb_mesh.clear()

        ply_paths = []
        for root, dirs, files in os.walk(self.base_dir):
            for fn in files:
                if fn.lower().endswith(".ply"):
                    ply_paths.append(os.path.join(root, fn))

        ply_paths.sort()
        for p in ply_paths:
            self.cmb_mesh.addItem(os.path.relpath(p, self.base_dir), userData=p)

        self.cmb_mesh.blockSignals(False)
        self.btn_show.setEnabled(self.cmb_mesh.count() > 0)

    def ensure_in_combo(self, path: str):
        if not path:
            return

        for i in range(self.cmb_mesh.count()):
            if self.cmb_mesh.itemData(i) == path:
                self.cmb_mesh.setCurrentIndex(i)
                self.btn_show.setEnabled(True)
                return

        label = os.path.relpath(path, self.base_dir) if os.path.commonpath([self.base_dir, os.path.abspath(path)]) == self.base_dir else os.path.basename(path)
        self.cmb_mesh.addItem(label, userData=path)
        self.cmb_mesh.setCurrentIndex(self.cmb_mesh.count() - 1)
        self.btn_show.setEnabled(True)

    def choose_mesh(self):

        filters = "PLY Mesh (*.ply);;Alle Dateien (*)"
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Mesh wählen", self.base_dir, filters)
        if not path:
            return
        self.ensure_in_combo(path)

    def pick_video(self):
        filters = ("Video-Dateien (*.mp4 *.MP4 *.mov *.MOV *.m4v *.M4V *.avi *.AVI *.mkv *.MKV *.webm *.WEBM *.wmv *.WMV);;"
                   "Alle Dateien (*)")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Video wählen", self.base_dir, filters)
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

    def compute(self):
        if not self.video_path:
            QtWidgets.QMessageBox.warning(self, "Hinweis", "Bitte Video auswählen.")
            return
        project = self.txt_project.text().strip() or "default_project"
        target_frames = int(self.spin_frames.value())
        self.btn_compute.setEnabled(False)
        self.worker = Worker(self.base_dir, self.video_path, project, target_frames)
        self.worker.progress.connect(self.set_progress)
        self.worker.log.connect(self.append_log)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()
        self.append_log("[ui] Pipeline gestartet…")

    def on_done(self, ply_path: str, project_root: str):
        self.last_ply = ply_path
        self.project_root = project_root
        self.btn_compute.setEnabled(True)
        self.btn_show.setEnabled(True)
        self.append_log(f"[ui] Fertig: {ply_path}")
        self.scan_meshes()
        if os.path.isfile(ply_path):
            self.ensure_in_combo(ply_path)

    def on_failed(self, msg: str):
        self.btn_compute.setEnabled(True)
        self.append_log("[error]\n" + msg)
        QtWidgets.QMessageBox.critical(self, "Fehler", msg)

    def show_mesh(self):
        ply_path = self.last_ply
        if self.cmb_mesh.currentIndex() >= 0:
            sel = self.cmb_mesh.currentData()
            if sel and os.path.isfile(sel):
                ply_path = sel

        if not ply_path or not os.path.isfile(ply_path):
            QtWidgets.QMessageBox.information(self, "Hinweis", "Kein gültiges PLY ausgewählt/gefunden.")
            return
        try:
            pcd = o3d.io.read_point_cloud(ply_path)
            o3d.visualization.draw_geometries([pcd])
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Fehler beim Öffnen", str(e))


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    mw = MainWindow()
    mw.show()
    app.exec()
