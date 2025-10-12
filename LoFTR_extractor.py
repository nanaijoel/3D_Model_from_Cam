import os
import cv2
import torch
import numpy as np
from typing import Optional, Tuple
from kornia.feature import LoFTR

# ==========================================================
#  GPU-beschleunigter LoFTR Extractor (outdoor/indoor)
#  -> unterstützt optionale Masken je Bild
# ==========================================================

def _resize_keep_aspect(img: np.ndarray, max_size: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale

def _apply_mask_u8(gray: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """ wendet 0/255-Maske auf Graubild an (Größe wird angepasst) """
    if mask is None:
        return gray
    if mask.shape[:2] != gray.shape[:2]:
        mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.bitwise_and(gray, gray, mask=mask)

class LoFTRExtractor:
    def __init__(self, device: str = "cuda", weights: str = "outdoor"):
        """
        device: 'cuda' oder 'cpu'
        weights: 'outdoor' (default) oder 'indoor'
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = LoFTR(pretrained=weights).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def match_pair(
        self,
        img1_path: str,
        img2_path: str,
        max_size: int = 640,
        mask1_path: Optional[str] = None,
        mask2_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gibt Pixelmatches (xy) in Originalbild-Koordinaten zurück.
        Masken (0/255) werden – falls vorhanden – auf die Bilder angewandt.
        """
        g1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        g2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        if g1 is None or g2 is None:
            raise FileNotFoundError(f"Could not read one of the images: {img1_path}, {img2_path}")

        m1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE) if mask1_path and os.path.isfile(mask1_path) else None
        m2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE) if mask2_path and os.path.isfile(mask2_path) else None

        # gleiche Szene wie LG/DISK: Maske anwenden
        g1m = _apply_mask_u8(g1, m1)
        g2m = _apply_mask_u8(g2, m2)

        g1s, s1 = _resize_keep_aspect(g1m, max_size)
        g2s, s2 = _resize_keep_aspect(g2m, max_size)

        t1 = torch.from_numpy(g1s)[None, None].float().to(self.device) / 255.0
        t2 = torch.from_numpy(g2s)[None, None].float().to(self.device) / 255.0

        out = self.model({"image0": t1, "image1": t2})
        if len(out["keypoints0"]) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

        pts0 = (out["keypoints0"].cpu().numpy().astype(np.float32)) / s1
        pts1 = (out["keypoints1"].cpu().numpy().astype(np.float32)) / s2
        return pts0, pts1
