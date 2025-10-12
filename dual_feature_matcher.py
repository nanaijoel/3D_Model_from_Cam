# dual_feature_matcher.py
# Zweck: Nur noch zum (optionalen) Debuggen – KEINE Fusion mehr.
# Schreibt je Algorithmus eigene Paar-Matches als .npz (pts0/pts1 in Pixelkoordinaten).
# Wenn ihr dieses File nicht braucht, könnt ihr es auch weglassen. Es stört die neue Pipeline nicht.

from pathlib import Path
import numpy as np
import json

def save_pair_matches_npz(out_dir: Path, algo: str, i: int, j: int, pts0: np.ndarray, pts1: np.ndarray):
    """
    Speichert ein Bildpaar-Match als .npz (xy, xy) in Pixelkoordinaten.
    Structure: matches/<algo>/{i}_{j}.npz mit keys: pts0, pts1
    """
    out_dir = Path(out_dir) / "matches" / algo
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{i}_{j}.npz", pts0=pts0.astype(np.float32), pts1=pts1.astype(np.float32))

def write_pairs_list(out_root: Path, pairs):
    """
    Schreibt eine gemeinsame Liste aller Bildpaare, die BEIDE Algorithmen verwenden.
    Format: JSON mit [[i,j], ...]
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "pairs.json", "w", encoding="utf-8") as f:
        json.dump([[int(a), int(b)] for (a,b) in pairs], f)

# Beispiel-Aufruf (nur Doku):
# save_pair_matches_npz("output", "loftr", 3, 4, pts0, pts1)
# save_pair_matches_npz("output", "disk",  3, 4, pts0, pts1)
# write_pairs_list("output", [(3,4), (4,5)])
