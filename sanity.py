import numpy as np

d2 = np.load("DHP19/data_2d_dhp19_gt.npz", allow_pickle=True)
meta = d2["metadata"].item()
pos2 = d2["positions_2d"].item()

print("num_joints:", meta["num_joints"])
s = list(pos2.keys())[0]
a = list(pos2[s].keys())[0]
print("cams:", len(pos2[s][a]), "shape:", pos2[s][a][0].shape)
