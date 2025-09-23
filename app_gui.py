# pip install PySide6 opencv-python opencv-contrib-python open3d numpy
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
        self.setWindowTitle("Video → 3D Reconstruction (modular)")
        self.resize(980, 540)

        self.video_path = ""
        self.base_dir = os.path.abspath(".")
        self.last_ply = ""
        self.project_root = ""

        # --- Controls ---
        self.btn_pick = QtWidgets.QPushButton("Video auswählen…")
        self.lbl_video = QtWidgets.QLabel("Keine Datei gewählt"); self.lbl_video.setWordWrap(True)

        self.txt_project = QtWidgets.QLineEdit(); self.txt_project.setPlaceholderText("Projektname (z.B. buddha_better)")
        self.spin_frames = QtWidgets.QSpinBox(); self.spin_frames.setRange(50, 2000); self.spin_frames.setValue(200); self.spin_frames.setSuffix(" Frames")

        self.btn_compute = QtWidgets.QPushButton("Compute")
        self.btn_compute.setEnabled(False)

        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
        self.lbl_stage = QtWidgets.QLabel("…")
        self.chk_autoscroll = QtWidgets.QCheckBox("Autoscroll Log"); self.chk_autoscroll.setChecked(True)

        self.log = QtWidgets.QTextEdit(); self.log.setReadOnly(True)
        self.btn_show = QtWidgets.QPushButton("Show 3D Model as Mesh"); self.btn_show.setEnabled(False)

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow(self.btn_pick, self.lbl_video)
        form.addRow("Projektname:", self.txt_project)
        form.addRow("Ziel-Frames:", self.spin_frames)
        form.addRow(self.btn_compute, self.progress)
        form.addRow("Schritt:", self.lbl_stage)
        form.addRow(self.chk_autoscroll)
        top = QtWidgets.QWidget(); top.setLayout(form)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(top); splitter.addWidget(self.log); splitter.setStretchFactor(1,1)

        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(bottom)
        bl.addWidget(self.btn_show)
        container = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(container)
        v.addWidget(splitter)
        v.addWidget(bottom)

        self.setCentralWidget(container)

        # Signals
        self.btn_pick.clicked.connect(self.pick_video)
        self.btn_compute.clicked.connect(self.compute)
        self.btn_show.clicked.connect(self.show_mesh)

        self.worker = None

    def pick_video(self):
        filters = "Video-Dateien (*.mp4 *.MP4 *.mov *.MOV *.m4v *.M4V *.avi *.AVI *.mkv *.MKV *.webm *.WEBM *.wmv *.WMV);;Alle Dateien (*)"
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Video wählen", "", filters)

        if not path: return
        self.video_path = path; self.lbl_video.setText(path)
        self.btn_compute.setEnabled(True); self.progress.setValue(0); self.log.clear(); self.btn_show.setEnabled(False)

    def append_log(self, msg: str):
        self.log.append(msg)
        if self.chk_autoscroll.isChecked():
            self.log.moveCursor(QTextCursor.End)
            self.log.ensureCursorVisible()

    def set_progress(self, val: int, label: str):
        self.progress.setValue(val); self.lbl_stage.setText(label)

    def compute(self):
        if not self.video_path:
            QtWidgets.QMessageBox.warning(self, "Hinweis", "Bitte Video auswählen."); return
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
        self.last_ply = ply_path; self.project_root = project_root
        self.btn_compute.setEnabled(True); self.btn_show.setEnabled(True)
        self.append_log(f"[ui] Fertig: {ply_path}")

    def on_failed(self, msg: str):
        self.btn_compute.setEnabled(True); self.append_log("[error]\n"+msg)
        QtWidgets.QMessageBox.critical(self, "Fehler", msg)

    def show_mesh(self):
        if not self.last_ply or not os.path.isfile(self.last_ply):
            QtWidgets.QMessageBox.information(self, "Hinweis", "Keine PLY gefunden."); return
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(self.last_ply)
        o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    mw = MainWindow(); mw.show(); app.exec()