import torch
import cv2
import numpy as np
from kornia.feature import LoFTR
from kornia_moons.feature import draw_LAF_matches

# ==========================================================
#  GPU-beschleunigter LoFTR Extractor (outdoor pretrained)
# ==========================================================

class LoFTRExtractor:
    def __init__(self, device: str = "cuda", weights: str = "inddoor"):
        """
        device: 'cuda' oder 'cpu'
        weights: 'outdoor' (default) oder 'indoor'
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = LoFTR(pretrained=weights).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def match_pair(self, img1_path: str, img2_path: str, max_size: int = 640):
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            raise FileNotFoundError(f"Could not read one of the images: {img1_path}, {img2_path}")

        # ---- Downscale for memory safety ----
        def resize_keep_aspect(img, max_size):
            h, w = img.shape[:2]
            scale = max_size / max(h, w)
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            return img, scale

        img1, s1 = resize_keep_aspect(img1, max_size)
        img2, s2 = resize_keep_aspect(img2, max_size)

        t1 = torch.from_numpy(img1)[None, None].float() / 255.0
        t2 = torch.from_numpy(img2)[None, None].float() / 255.0
        t1, t2 = t1.to(self.device), t2.to(self.device)

        input_dict = {"image0": t1, "image1": t2}
        output = self.model(input_dict)

        if len(output["keypoints0"]) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))

        pts0 = output["keypoints0"].cpu().numpy() / s1
        pts1 = output["keypoints1"].cpu().numpy() / s2
        return pts0, pts1

    def visualize_matches(self, img1_path, img2_path, save_path="loftr_matches.jpg"):
        """
        Optionale Visualisierung der LoFTR-Matches.
        """
        pts0, pts1 = self.match_pair(img1_path, img2_path)
        if len(pts0) == 0:
            print("[LoFTR] Keine Matches gefunden.")
            return
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        matched_img = draw_LAF_matches(img1, img2, pts0, pts1, inliers=np.ones(len(pts0)))
        cv2.imwrite(save_path, matched_img)
        print(f"[LoFTR] Matches visualisiert unter {save_path}")

"""
if __name__ == "__main__":
    loftr = LoFTRExtractor(device="cuda", weights="outdoor")
    p1 = "projects/BUDDHA_AWESOME/raw_frames/frame_0001_src_000011.png"
    p2 = "projects/BUDDHA_AWESOME/raw_frames/frame_0002_src_000022.png"
    pts0, pts1 = loftr.match_pair(p1, p2)
    print("Gefundene Matches:", len(pts0))
"""

