import numpy as np
from common.skeleton import Skeleton

class Dhp19Dataset:
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self._pos = d["positions_3d"].item()
        self._cams = d["cameras"].item()

        # Only used for left/right indices + some viz.
        self._skeleton = Skeleton(
            parents=[-1] * 13,                 # parents not used for training here
            joints_left=[3, 5, 8, 10, 12],
            joints_right=[2, 4, 7, 9, 11],
        )

    def subjects(self): return list(self._pos.keys())
    def cameras(self):  return self._cams
    def skeleton(self): return self._skeleton
    def fps(self):      return 100
    def __getitem__(self, subject): return self._pos[subject]
