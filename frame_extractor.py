import os
import cv2 as cv
import numpy as np
from typing import Callable, List

def extract_and_save_frames(video_path: str, target_frames: int, out_dir: str,
                            on_log: Callable[[str], None] = None,
                            on_progress: Callable[[int,str], None] = None) -> List[str]:

    def log(m): on_log and on_log(m)
    def prog(p, s): on_progress and on_progress(p, s)

    cap = cv.VideoCapture(video_path)
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError("Konnte Frame-Anzahl nicht lesen.")
    n = max(3, int(target_frames))
    idxs = np.linspace(0, total-1, num=n).astype(int)

    saved = []
    for i, idx in enumerate(idxs):
        cap.set(cv.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            log(f"[frames] WARN: Frame {idx} nicht lesbar.")
            continue
        fn = os.path.join(out_dir, f"frame_{i:04d}_src_{idx:06d}.png")
        cv.imwrite(fn, frame)
        saved.append(fn)
        prog(int(5 + (i+1)/len(idxs)*25), "Frames extrahieren")
    cap.release()
    if len(saved) < 3:
        raise RuntimeError("Zu wenige gültige Frames nach Sampling.")
    log(f"[frames] gespeichert: {len(saved)}")
    return saved
