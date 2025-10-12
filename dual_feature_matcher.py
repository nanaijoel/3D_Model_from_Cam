import cv2
import torch
import numpy as np
from pathlib import Path
from LoFTR_extractor import LoFTRExtractor
from lightglue import LightGlue, DISK
from lightglue.utils import numpy_image_to_torch

# ==========================================================
#  Dual Feature Matcher: LoFTR + LightGlue(DISK)
# ==========================================================

class DualFeatureMatcher:
    def __init__(self, device: str = "cuda", loftr_weights="outdoor"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # LoFTR-Instanz (low-texture robust)
        self.loftr = LoFTRExtractor(device=device, weights=loftr_weights)

        # LightGlue mit DISK (high-texture robust)
        extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
        self.lightglue = LightGlue(features='disk').eval().to(self.device)
        self.extractor = extractor

    @torch.inference_mode()
    def match_pair(self, img1_path: str, img2_path: str, ransac=True, save_debug=None):
        """
        Kombiniert LoFTR + LightGlue Matches
        """
        # ---------- LoFTR ----------
        pts0_loftr, pts1_loftr = self.loftr.match_pair(img1_path, img2_path)
        n_loftr = len(pts0_loftr)

        # ---------- LightGlue ----------
        img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
        img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
        t1 = numpy_image_to_torch(img1)[None].to(self.device)
        t2 = numpy_image_to_torch(img2)[None].to(self.device)

        feats1 = self.extractor.extract(t1)
        feats2 = self.extractor.extract(t2)
        out = self.lightglue({"image0": feats1, "image1": feats2})

        # Extrahiere Matches robust für neue & alte Versionen
        if "matches0" in out and "matches1" in out:
            matches0 = out["matches0"][0].cpu().numpy()
            matches1 = out["matches1"][0].cpu().numpy()
            valid = matches0 > -1
            feats1_np = feats1["keypoints"][0].cpu().numpy()
            feats2_np = feats2["keypoints"][0].cpu().numpy()
            pts0_light = feats1_np[valid]
            pts1_light = feats2_np[matches0[valid]]
            n_lightglue = len(pts0_light)
        else:
            print("[WARN] LightGlue output format not recognized")
            pts0_light, pts1_light = np.zeros((0, 2)), np.zeros((0, 2))
            n_lightglue = 0

        # ---------- Fusion ----------
        fused_pts0 = np.concatenate([pts0_loftr, pts0_light], axis=0)
        fused_pts1 = np.concatenate([pts1_loftr, pts1_light], axis=0)

        # Duplikate (innerhalb 1 Pixel) entfernen
        all_matches = np.hstack([fused_pts0, fused_pts1])
        _, unique_idx = np.unique(all_matches.round().astype(int), axis=0, return_index=True)
        fused_pts0 = fused_pts0[unique_idx]
        fused_pts1 = fused_pts1[unique_idx]

        if ransac and len(fused_pts0) >= 8:
            F, inliers = cv2.findFundamentalMat(fused_pts0, fused_pts1, cv2.FM_RANSAC, 1.5, 0.99)
            fused_pts0 = fused_pts0[inliers.ravel() > 0]
            fused_pts1 = fused_pts1[inliers.ravel() > 0]

        print(f"[DualMatcher] LoFTR: {n_loftr}, LightGlue: {n_lightglue}, Fused: {len(fused_pts0)}")

        if save_debug:
            vis = self._draw_matches(img1, img2, fused_pts0, fused_pts1)
            cv2.imwrite(save_debug, vis)

        torch.cuda.empty_cache()
        return fused_pts0, fused_pts1

    def _draw_matches(self, img1, img2, pts0, pts1):
        """Visualisierung der kombinierten Matches"""
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


# ==========================================================
#  Testlauf
# ==========================================================
if __name__ == "__main__":
    matcher = DualFeatureMatcher(device="cuda")
    p1 = "projects/BUDDHA_AWESOME/raw_frames/frame_0001_src_000011.png"
    p2 = "projects/BUDDHA_AWESOME/raw_frames/frame_0002_src_000022.png"

    pts0, pts1 = matcher.match_pair(p1, p2, save_debug="dual_matches_debug.jpg")
    print("Finale kombinierte Matches:", len(pts0))
