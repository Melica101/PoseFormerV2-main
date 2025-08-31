import cdflib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cdf_file = cdflib.CDF('Directions.cdf')
gtPose3D = cdf_file['pose']

selected_joint_indices = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
selected_coord_indices = []
for joint_idx in selected_joint_indices:
    selected_coord_indices.extend([3 * joint_idx, 3 * joint_idx + 1, 3 * joint_idx + 2])

gtPose3D_selected = gtPose3D[:, :, selected_coord_indices]
print(gtPose3D_selected.shape)

data = np.load('demo/output/test2/predictions_3d.npz')
preds = data['predictions']
print(preds.shape)

preds = preds.reshape((2699, 17, 3))
preds = preds.reshape((2699, 51))
preds = preds[np.newaxis, :, :]

print("Final prediction shape:", preds.shape)
pred_xyz = preds[0].reshape(2699, 17, 3)
gt_xyz = gtPose3D_selected[0].reshape(2699, 17, 3)/1000

pred_aligned = pred_xyz - pred_xyz[:, [0], :]
gt_aligned = gt_xyz - gt_xyz[:, [0], :]

# R_x_90 = np.array([
#     [1, 0, 0],
#     [0, 0, -1],
#     [0, 1, 0]
# ])

# # Apply the rotation to your Nx3 joint coordinates
# pred_aligned = pred_aligned @ R_x_90.T

# R_z_180 = np.array([
#     [-1, 0, 0],
#     [ 0, -1, 0],
#     [ 0,  0, 1]
# ])
# pred_aligned = pred_aligned @ R_z_180.T
# pred_aligned[:, :, 1:] *= -1  # flips Y and Z
# pred_aligned = pred_aligned[:, :, [0, 2, 1]]
# pred_aligned[:, 2] *= -1
# rot_x_180 = np.array([
#     [1,  0,   0],
#     [0, -1,   0],
#     [0,  0,  -1]
# ])
# pred_aligned = pred_aligned @ rot_x_180.T
# rot_x_270 = np.array([
#     [1, 0,  0],
#     [0, 0,  1],
#     [0, -1, 0]
# ])

# pred_aligned = pred_aligned @ rot_x_270.T
# rot_x_90 = np.array([
#     [1, 0,  0],
#     [0, 0, -1],
#     [0, 1,  0]
# ])

# pred_aligned = pred_aligned @ rot_x_90.T
# rot_y_90 = np.array([
#     [0, 0, 1],
#     [0, 1, 0],
#     [-1, 0, 0]
# ])

# pred_aligned = pred_aligned @ rot_y_90.T
# rot_y_180 = np.array([
#     [-1, 0,  0],
#     [ 0, 1,  0],
#     [ 0, 0, -1]
# ])

# pred_aligned = pred_aligned @ rot_y_180.T
# rot_matrix = np.array([
#     [0, 0, 1],
#     [0, 1, 0],
#     [1, 0, 0]
# ])

# pred_aligned = pred_aligned @ rot_matrix.T

rot_z_90 = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])
rot_z_45 = np.array([
    [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
    [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
    [0, 0, 1]
])
rot_z_30 = np.array([
    [np.cos(np.pi / 6), -np.sin(np.pi / 6), 0],
    [np.sin(np.pi / 6), np.cos(np.pi / 6), 0],
    [0, 0, 1]
])
rot_z_60 = np.array([
    [np.cos(np.pi / 3), -np.sin(np.pi / 3), 0],
    [np.sin(np.pi / 3), np.cos(np.pi / 3), 0],
    [0, 0, 1]
])
rot_z_10 = np.array([
    [np.cos(np.radians(40)), -np.sin(np.radians(40)), 0],
    [np.sin(np.radians(40)), np.cos(np.radians(40)), 0],
    [0, 0, 1]
])
rot_z_neg_90 = np.array([
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 1]
])

rot_z_neg_30 = np.array([
    [np.cos(np.radians(-30)), -np.sin(np.radians(-30)), 0],
    [np.sin(np.radians(-30)), np.cos(np.radians(-30)), 0],
    [0, 0, 1]
])
# rot_z_neg_135 = np.array([
#     [np.cos(np.radians(-145)), -np.sin(np.radians(-145)), 0],
#     [np.sin(np.radians(-145)), np.cos(np.radians(-145)), 0],
#     [0, 0, 1]
# ])
rot_z_neg_135 = np.array([
    [np.cos(np.radians(-135)), -np.sin(np.radians(-135)), 0],
    [np.sin(np.radians(-135)), np.cos(np.radians(-135)), 0],
    [0, 0, 1]
])
# Apply the rotation (using matrix multiplication)
# Apply the rotation (using matrix multiplication)
# pred_aligned = pred_aligned @ rot_z_neg_90.T

# # Apply the rotation (using matrix multiplication)
pred_aligned = pred_aligned @ rot_z_neg_135.T

errors = np.linalg.norm(pred_aligned - gt_aligned, axis=2)  # shape: (2699, 17)

mpjpe = np.mean(errors)
print(f"MPJPE: {mpjpe:.2f} mm")

threshold = 0.5
correct_keypoints = errors < threshold
total_keypoints = errors.size

num_correct = np.sum(correct_keypoints)
pck = 100.0 * num_correct / total_keypoints
print(f"PCK @ {threshold:.0f}mm: {pck:.2f}%")

# def plot_pose(joints, title):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2])
#     ax.set_title(title)
#     plt.show()

# plot_pose(gt_xyz[0], "Ground Truth Full Pose")
# plot_pose(pred_xyz[0], "Prediction")
# plot_pose(gt_aligned[0], "Ground Truth")

# data3d = np.load('data/data_3d_h36m.npz', allow_pickle=True)
# data3d = data3d['positions_3d'].item()

# data3d = data3d['S9']['Directions'][:, selected_joint_indices, :]

# data3d = data3d.reshape(1, 2699, 51) 

# print(data3d.shape)

# data_xyz = data3d[0].reshape(2699, 17, 3)

# data_aligned = data_xyz - data_xyz[:, [0], :]

# errors2 = np.linalg.norm(pred_aligned - data_aligned, axis=2)

# mpjpe2 = np.mean(errors2)
# print(f"MPJPE: {mpjpe2:.2f} mm")
# print(pred_aligned[2], data_aligned[2])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    max_range = max(x_range, y_range, z_range)

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

def plot_pose(joints, title="Pose", elev=0, azim=-90):
    """Plot 3D pose centered at the hip (joint 0)."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    joints = joints - joints[0]  # make hip joint the origin
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='b', marker='o')

    skeleton = [
        (0, 1), (1, 2), (2, 3),         # Right leg
        (0, 4), (4, 5), (5, 6),         # Left leg
        (0, 7), (7, 8), (8, 9), (9,10), # Spine and head
        (8,11), (11,12), (12,13),       # Left arm
        (8,14), (14,15), (15,16)        # Right arm
    ]
    for i, j in skeleton:
        if i < len(joints) and j < len(joints):
            ax.plot([joints[i, 0], joints[j, 0]],
                    [joints[i, 1], joints[j, 1]],
                    [joints[i, 2], joints[j, 2]], 'k-')

    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)  # camera angle
    set_axes_equal(ax)
    plt.show()
    
# def plot_two_poses(gt, pred, frame_idx=0, elev=70, azim=-90):
#     joints_gt = gt[frame_idx] - gt[frame_idx][0]  # center on hip
#     joints_pred = pred[frame_idx] - pred[frame_idx][0]

#     skeleton = [
#         (0, 1), (1, 2), (2, 3),
#         (0, 4), (4, 5), (5, 6),
#         (0, 7), (7, 8), (8, 9), (9,10),
#         (8,11), (11,12), (12,13),
#         (8,14), (14,15), (15,16)
#     ]

#     fig = plt.figure(figsize=(12, 6))

#     ax1 = fig.add_subplot(1, 2, 1, projection='3d')
#     ax1.set_title("Ground Truth")
#     ax1.scatter(joints_gt[:, 0], joints_gt[:, 1], joints_gt[:, 2], c='g', marker='o')
#     for i, j in skeleton:
#         ax1.plot([joints_gt[i, 0], joints_gt[j, 0]],
#                  [joints_gt[i, 1], joints_gt[j, 1]],
#                  [joints_gt[i, 2], joints_gt[j, 2]], 'k-')
#     # ax1.view_init(elev=elev, azim=azim)
#     set_axes_equal(ax1)
#     ax1.view_init(elev=elev, azim=azim)  # <-- Also after limits


#     ax2 = fig.add_subplot(1, 2, 2, projection='3d')
#     ax2.set_title("Prediction")
#     ax2.scatter(joints_pred[:, 0], joints_pred[:, 1], joints_pred[:, 2], c='b', marker='o')
#     for i, j in skeleton:
#         ax2.plot([joints_pred[i, 0], joints_pred[j, 0]],
#                  [joints_pred[i, 1], joints_pred[j, 1]],
#                  [joints_pred[i, 2], joints_pred[j, 2]], 'k-')
#     # ax2.view_init(elev=elev, azim=azim)
#     set_axes_equal(ax2)
#     ax2.view_init(elev=elev, azim=azim)  # <-- Also after limits


#     plt.tight_layout()
#     plt.show()

# plot_pose(data_aligned[1], "Data3D Full Pose")
# plot_pose(pred_aligned[1], "Prediction Aligned")

# def plot_two_poses(gt, pred, frame_idx=0, elev=10, azim=-70):
#     joints_gt = gt[frame_idx] - gt[frame_idx][0]  # center on hip
#     joints_pred = pred[frame_idx] - pred[frame_idx][0]

#     # --- Normalize scale ---
#     # Get max distance from origin (hip) for scale
#     scale_gt = np.max(np.linalg.norm(joints_gt, axis=1))
#     scale_pred = np.max(np.linalg.norm(joints_pred, axis=1))
#     max_scale = max(scale_gt, scale_pred)

#     joints_gt /= max_scale
#     joints_pred /= max_scale

#     # --- Shared axis limits ---
#     all_joints = np.concatenate([joints_gt, joints_pred], axis=0)
#     xyz_min = all_joints.min(axis=0)
#     xyz_max = all_joints.max(axis=0)

#     buffer = 0.1
#     limits = [(xyz_min[i] - buffer, xyz_max[i] + buffer) for i in range(3)]

#     skeleton = [
#         (0, 1), (1, 2), (2, 3),
#         (0, 4), (4, 5), (5, 6),
#         (0, 7), (7, 8), (8, 9), (9,10),
#         (8,11), (11,12), (12,13),
#         (8,14), (14,15), (15,16)
#     ]

#     fig = plt.figure(figsize=(12, 6))

#     def plot_pose(ax, joints, color, title):
#         ax.set_title(title)
#         ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, marker='o')
#         for i, j in skeleton:
#             ax.plot([joints[i, 0], joints[j, 0]],
#                     [joints[i, 1], joints[j, 1]],
#                     [joints[i, 2], joints[j, 2]], 'k-')
#         ax.view_init(elev=elev, azim=azim)
#         ax.set_xlim(limits[0])
#         ax.set_ylim(limits[1])
#         ax.set_zlim(limits[2])

#     ax1 = fig.add_subplot(1, 2, 1, projection='3d')
#     plot_pose(ax1, joints_gt, 'g', "Ground Truth")

#     ax2 = fig.add_subplot(1, 2, 2, projection='3d')
#     plot_pose(ax2, joints_pred, 'b', "Prediction")

#     plt.tight_layout()
#     plt.show()

# plot_two_poses(gt_aligned, pred_aligned, frame_idx=0)
# def plot_two_poses(gt, pred, frame_idx=0, elev=10, azim=-70):
#     joints_gt = gt[frame_idx] - gt[frame_idx][0]  # center on hip
#     joints_pred = pred[frame_idx] - pred[frame_idx][0]

#     # --- Normalize scale ---
#     # Get max distance from origin (hip) for scale
#     scale_gt = np.max(np.linalg.norm(joints_gt, axis=1))
#     scale_pred = np.max(np.linalg.norm(joints_pred, axis=1))
#     max_scale = max(scale_gt, scale_pred)

#     joints_gt /= max_scale
#     joints_pred /= max_scale

#     # --- Shared axis limits ---
#     all_joints = np.concatenate([joints_gt, joints_pred], axis=0)
#     xyz_min = all_joints.min(axis=0)
#     xyz_max = all_joints.max(axis=0)

#     buffer = 0.1
#     limits = [(xyz_min[i] - buffer, xyz_max[i] + buffer) for i in range(3)]

#     skeleton = [
#         (0, 1), (1, 2), (2, 3),
#         (0, 4), (4, 5), (5, 6),
#         (0, 7), (7, 8), (8, 9), (9,10),
#         (8,11), (11,12), (12,13),
#         (8,14), (14,15), (15,16)
#     ]

#     fig = plt.figure(figsize=(12, 6))

#     def plot_pose(ax, joints_gt, joints_pred, color_gt, color_pred, title):
#         ax.set_title(title)
#         ax.scatter(joints_gt[:, 0], joints_gt[:, 1], joints_gt[:, 2], c=color_gt, marker='o')
#         ax.scatter(joints_pred[:, 0], joints_pred[:, 1], joints_pred[:, 2], c=color_pred, marker='o')
#         for i, j in skeleton:
#             ax.plot([joints_gt[i, 0], joints_gt[j, 0]], [joints_gt[i, 1], joints_gt[j, 1]], [joints_gt[i, 2], joints_gt[j, 2]], color=color_gt)
#             ax.plot([joints_pred[i, 0], joints_pred[j, 0]], [joints_pred[i, 1], joints_pred[j, 1]], [joints_pred[i, 2], joints_pred[j, 2]], color=color_pred)
#         ax.view_init(elev=elev, azim=azim)
#         ax.set_xlim(limits[0])
#         ax.set_ylim(limits[1])
#         ax.set_zlim(limits[2])

#     ax1 = fig.add_subplot(1, 1, 1, projection='3d')
#     plot_pose(ax1, joints_gt, joints_pred, 'g', 'b', "Ground Truth (green) vs Prediction (blue)")

#     plt.tight_layout()
#     plt.show()

# plot_two_poses(gt_aligned, pred_aligned, frame_idx=0)
import matplotlib.pyplot as plt
import numpy as np

def set_axes_equal(ax):
    """Set the 3D axes to have equal scaling"""
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()
    
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    
    max_range = max(x_range, y_range, z_range)
    
    mid_x = (x_limits[1] + x_limits[0]) / 2
    mid_y = (y_limits[1] + y_limits[0]) / 2
    mid_z = (z_limits[1] + z_limits[0]) / 2
    
    ax.set_xlim([mid_x - max_range / 2, mid_x + max_range / 2])
    ax.set_ylim([mid_y - max_range / 2, mid_y + max_range / 2])
    ax.set_zlim([mid_z - max_range / 2, mid_z + max_range / 2])

def plot_two_poses(gt, pred, frame_idx=0, elev=20, azim=-70):
    # Center the joints on the first joint (usually the hip)
    joints_gt = gt[frame_idx] - gt[frame_idx][0]  # center on hip
    joints_pred = pred[frame_idx] - pred[frame_idx][0]

    # Define the skeleton connections (pairs of joints)
    skeleton = [
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9), (9, 10),
        (8, 11), (11, 12), (12, 13),
        (8, 14), (14, 15), (15, 16)
    ]
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title("Ground Truth and Prediction")

    # Plot the ground truth skeleton (green points and black lines)
    ax.scatter(joints_gt[:, 0], joints_gt[:, 1], joints_gt[:, 2], c='g', marker='o', label='Ground Truth')
    for i, j in skeleton:
        ax.plot([joints_gt[i, 0], joints_gt[j, 0]],
                [joints_gt[i, 1], joints_gt[j, 1]],
                [joints_gt[i, 2], joints_gt[j, 2]], 'k-')

    # Plot the predicted skeleton (blue points and red dashed lines)
    ax.scatter(joints_pred[:, 0], joints_pred[:, 1], joints_pred[:, 2], c='b', marker='o', label='Prediction')
    for i, j in skeleton:
        ax.plot([joints_pred[i, 0], joints_pred[j, 0]],
                [joints_pred[i, 1], joints_pred[j, 1]],
                [joints_pred[i, 2], joints_pred[j, 2]], 'r--')  # Dashed red lines for prediction

    # Set the view angle and make axes equal
    ax.view_init(elev=elev, azim=azim)
    set_axes_equal(ax)

    # Add a legend to distinguish the ground truth and predicted points
    ax.legend()

    plt.tight_layout()
    plt.show()

# Assuming gt_aligned and pred_aligned are defined and contain the skeleton data
plot_two_poses(gt_aligned, pred_aligned, frame_idx=2500)
