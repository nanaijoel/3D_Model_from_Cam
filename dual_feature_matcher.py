import os
import cv2
import torch
import numpy as np
from typing import Optional
from LoFTR_extractor import LoFTRExtractor
from lightglue import LightGlue, DISK
from lightglue.utils import numpy_image_to_torch

# ==========================================================
#  Dual Feature Matcher: LoFTR + LightGlue(DISK) (pixelbasiert)
#  – nutzt Masken für BEIDE Pfade
#  – NUR für Debug/Visualisierung; für SfM nimm image_matching.py
# ==========================================================

def _mask_for(img_path: str, mask_dir: str) -> Optional[str]:
    base = os.path.splitext(os.path.basename(img_path))[0]
    mp = os.path.join(mask_dir, f"{base}_mask.png")
    return mp if os.path.isfile(mp) else None

class DualFeatureMatcher:
    def __init__(self, device: str = "cuda", loftr_weights="outdoor", mask_dir: Optional[str] = None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.loftr = LoFTRExtractor(device=device, weights=loftr_weights)

        # LightGlue + DISK
        self.extractor = DISK(max_num_keypoints=int(os.getenv("DFM_MAX_KP", "2048"))).eval().to(self.device)
        self.lightglue = LightGlue(features='disk').eval().to(self.device)
        self.mask_dir = mask_dir

    @torch.inference_mode()
    def match_pair(self, img1_path: str, img2_path: str, ransac=True, save_debug=None):
        # ---------- LoFTR (mit Masken) ----------
        m1 = _mask_for(img1_path, self.mask_dir) if self.mask_dir else None
        m2 = _mask_for(img2_path, self.mask_dir) if self.mask_dir else None
        pts0_loftr, pts1_loftr = self.loftr.match_pair(img1_path, img2_path, max_size=int(os.getenv("LOFTR_MAX_SIZE", "640")), mask1_path=m1, mask2_path=m2)
        n_loftr = len(pts0_loftr)

        # ---------- LightGlue ----------
        img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
        img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
        t1 = numpy_image_to_torch(img1)[None].to(self.device)
        t2 = numpy_image_to_torch(img2)[None].to(self.device)

        feats1 = self.extractor.extract(t1)
        feats2 = self.extractor.extract(t2)
        out = self.lightglue({"image0": feats1, "image1": feats2})

        if "matches0" in out and "matches1" in out:
            matches0 = out["matches0"][0].cpu().numpy()
            valid = matches0 > -1
            k1 = feats1["keypoints"][0].cpu().numpy()
            k2 = feats2["keypoints"][0].cpu().numpy()
            pts0_light = k1[valid]
            pts1_light = k2[matches0[valid]]
            n_lightglue = len(pts0_light)
        else:
            pts0_light = np.zeros((0, 2), np.float32)
            pts1_light = np.zeros((0, 2), np.float32)
            n_lightglue = 0

        # ---------- Fusion ----------
        fused_pts0 = np.concatenate([pts0_loftr, pts0_light], axis=0)
        fused_pts1 = np.concatenate([pts1_loftr, pts1_light], axis=0)

        # Duplikate (≈1px) entfernen
        if len(fused_pts0):
            all_matches = np.hstack([fused_pts0, fused_pts1])
            _, unique_idx = np.unique(all_matches.round().astype(int), axis=0, return_index=True)
            fused_pts0 = fused_pts0[unique_idx]
            fused_pts1 = fused_pts1[unique_idx]

        if ransac and len(fused_pts0) >= 8:
            _, inl = cv2.findFundamentalMat(fused_pts0, fused_pts1, cv2.FM_RANSAC, 1.5, 0.99)
            if inl is not None:
                inl = inl.ravel() > 0
                fused_pts0 = fused_pts0[inl]
                fused_pts1 = fused_pts1[inl]

        print(f"[DualMatcher] LoFTR: {n_loftr}, LightGlue: {n_lightglue}, Fused: {len(fused_pts0)}")

        if save_debug:
            vis = self._draw_matches(img1, img2, fused_pts0, fused_pts1)
            cv2.imwrite(save_debug, vis)

        torch.cuda.empty_cache()
        return fused_pts0, fused_pts1

    def _draw_matches(self, img1, img2, pts0, pts1):
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1
        canvas[:h2, w1:w1 + w2] = img2

        for p0, p1 in zip(pts0, pts1):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            p1_shifted = (int(p1[0] + w1), int(p1[1]))
            cv2.line(canvas, tuple(np.int32(p0)), p1_shifted, color, 1, cv2.LINE_AA)
        return canvas

if __name__ == "__main__":
    # Beispiel (Passe die Pfade an dein Projekt an)
    proj = "projects/default_project"
    img_dir = os.path.join(proj, "raw_frames")
    mask_dir = os.path.join(proj, "features", "masks")

    imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
    matcher = DualFeatureMatcher(device="cuda", loftr_weights="outdoor", mask_dir=mask_dir)
    if len(imgs) >= 2:
        matcher.match_pair(imgs[0], imgs[1], save_debug=os.path.join(proj, "dual_matches_debug.jpg"))
