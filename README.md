# 3D_Model_from_Cam 
_Reconstruct high-quality 3D point clouds from a short handheld video._

This repository implements a modular **Structure-from-Motion (SfM)** and **Multi-View Stereo (MVS)** pipeline with automatic background masking, LightGlue feature matching.  
A simple GUI application is added, so the whole program can be run by just execute the app_gui.py.
The configuration is fully controlled via `config.yaml`, which maps directly to environment variables.

---

## Features

- **Automatic frame extraction** from video input  
- **Image preprocessing** (CLAHE, unsharp masking, etc.)  
- **Masking with rembg for background and foreground mask 
- **Feature extraction & matching** using [DISK](https://github.com/cvlab-epfl/disk) + [LightGlue](https://github.com/cvg/LightGlue)  
- **Incremental SfM** (camera poses, sparse reconstruction)  
- **Optional dense MVS refinement**  
- **Automatic object masking** using `rembg`  
- **Optional texturing and visualization**  
- **Simple GUI** built with PySide6  

---

## Repository Structure

| File / Folder | Description |
|----------------|-------------|
| `app_gui.py` | PySide6-based GUI for interactive reconstruction |
| `pipeline_runner.py` | Main orchestrator for the full processing pipeline |
| `frame_extractor.py` | Extracts evenly spaced frames from input videos |
| `feature_extraction.py` | Extracts and preprocesses local image features |
| `image_matching.py` | Performs feature matching between frame pairs |
| `sfm_incremental.py` | Core incremental structure-from-motion logic |
| `meshing.py` | Dense depth fusion and optional mesh generation |
| `texturing.py` | Adds photometric texture to fused point clouds |
| `image_masking.py` | Handles background removal (e.g., via `rembg`) |
| `camera_pose_plot.py` | Visualizes camera trajectories (3D plot) |
| `config.yaml` | Global configuration file (auto-mapped to ENV vars) |

---

## Installation

```bash
# Clone repository
git clone https://github.com/nanaijoel/3D_Model_from_Cam.git
cd 3d-reconstruction-pipeline

# Create environment (optional)
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage
1. Make a video of an object in by moving the camera around it by 360 degrees.
   Good light conditions and a good cameras are recommended (smartphone camera is enough).
2. When the video is available on your computer, start the program by executing app_gui.py.
3. Select your video by button "Choose video"
4. Define the amount of frames which should be made for the process. 
   For e.g. 120 frames is recommended for a 360 degree video which was made in 30s.
5. Press compute - process begins.
6. When its completed, you will find all created point cloud files in the dropdown menu below.
7. Select a pyl file and look observe after pressing "Open 3D Model".
   Most likely you want to check the fused_textured_points.ply or fused_points.ply.
8. For Linux users with an installed version MeshLab2025.07-linux_x86_64, the button "Open MeshLab" is executebale.
   You can modify the point cloud there as needed.
