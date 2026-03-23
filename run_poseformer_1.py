# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Modified by Qitao Zhao (qitaozhao@mail.sdu.edu.cn)

import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import math
import logging
import traceback

from einops import rearrange, repeat
from copy import deepcopy
from tqdm import tqdm

from common.camera import *
import collections

from common.model_poseformer import *

from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import *

def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = inputs_3d.permute(1,0,2,3)
    out_num = inputs_2d_p.shape[0] - receptive_field + 1
    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    for i in range(out_num):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
    return eval_input_2d, inputs_3d_p

if __name__ == '__main__':
    args = parse_args()
    log =  logging.getLogger()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''.join(args.gpu)

    try:
        # Create checkpoint directory if it does not exist
        os.makedirs(args.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

    print('Loading dataset...', flush=True)
    dataset_path = 'data/data_3d_' + args.dataset + '.npz'
    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path)
    elif args.dataset == 'dhp19':
        from common.dhp19_dataset import Dhp19Dataset
        dataset = Dhp19Dataset('data/data_3d_dhp19.npz')
    elif args.dataset.startswith('custom'):
        from common.custom_dataset import CustomDataset
        dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
    else:
        raise KeyError('Invalid dataset')

    print('Preparing data...', flush=True)
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d
            elif 'positions_3d' in anim and (args.dataset == 'dhp19' or args.dataset.startswith('custom')):
                # For DHP19: normalize 3D coordinates to match model's output scale
                positions_3d_list = anim['positions_3d']
                for i in range(len(positions_3d_list)):
                    pos_3d = positions_3d_list[i]
                    # Ensure root-relative coordinates
                    pos_3d[:, 1:] -= pos_3d[:, :1] 
                    if args.dataset == 'dhp19':
                        pos_3d_normalized = pos_3d / 100.0  # mm -> normalized
                        positions_3d_list[i] = pos_3d_normalized
                anim['positions_3d'] = positions_3d_list

    print('Loading 2D detections...', flush=True)
    keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
    print("loaded keypoints from", 'data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', flush=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()

    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue
            for cam_idx in range(len(keypoints[subject][action])):
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]
            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                if args.std != 0:
                    kps += np.random.normal(loc=0.0, scale=args.std, size=kps.shape)
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    if args.dataset == 'h36m':
        subjects_train = 'S1,S5,S6,S7,S8'.split(',')
        subjects_test = 'S9,S11'.split(',')
    elif args.dataset == 'dhp19':
        subjects_train = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12']
        subjects_test = ['S13', 'S14', 'S15', 'S16', 'S17']
    else:
        subjects_train = args.subjects_train.split(',')
        subjects_test = args.subjects_test.split(',')

    def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True, load_gt=False):
        out_poses_3d = []
        out_poses_2d = []
        out_camera_params = []
        for subject in subjects:
            for action in keypoints[subject].keys():
                if action_filter is not None:
                    found = False
                    for a in action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue
                poses_2d = keypoints[subject][action]
                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)):
                        if np.isnan(poses_3d[i]).any() or np.isnan(poses_2d[i]).any():
                            continue
                        out_poses_3d.append(poses_3d[i])
                        out_poses_2d.append(poses_2d[i])
                        if subject in dataset.cameras():
                             cams = dataset.cameras()[subject]
                             if 'intrinsic' in cams[i]:
                                 out_camera_params.append(cams[i]['intrinsic'])
                else:
                    for i in range(len(poses_2d)):
                         out_poses_2d.append(poses_2d[i])
                         if subject in dataset.cameras():
                             cams = dataset.cameras()[subject]
                             if 'intrinsic' in cams[i]:
                                 out_camera_params.append(cams[i]['intrinsic'])
        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None
        stride = args.downsample
        if subset < 1:
            for i in range(len(out_poses_2d)):
                n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
                start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
                out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
        elif stride > 1:
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]
        return out_camera_params, out_poses_3d, out_poses_2d

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        print('Selected actions:', action_filter)

    cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)
    print(f'INFO: Loaded {len(poses_valid) if poses_valid is not None else 0} valid sequences', flush=True)

    receptive_field = args.number_of_frames
    print('INFO: Receptive field: {} frames'.format(receptive_field))
    pad = (receptive_field -1) // 2 
    min_loss = 100000
    cam = dataset.cameras()[subjects_test[0]][0] if dataset.cameras() else {'res_w': 346, 'res_h': 260}
    width = cam['res_w']
    height = cam['res_h']
    num_joints = keypoints_metadata['num_joints']

    num_joints_out = num_joints
    model_embed_dim = None
    joint_mapping = None

    if args.dataset == 'dhp19' and (args.resume or args.evaluate):
        # Force embed_dim=544 to match H36M checkpoint (17 joints * 32 embed_dim_ratio = 544)
        model_num_joints = 17
        num_joints_out = 13
        model_embed_dim = 544
        # DHP19 to H36M semantic mapping
        joint_mapping = [10, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]

    model_pos_train = PoseTransformerV2(num_frame=receptive_field, num_joints=model_num_joints, in_chans=2, embed_dim=model_embed_dim, joint_mapping=joint_mapping,
            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1, args=args, num_joints_out=num_joints_out)

    model_pos = PoseTransformerV2(num_frame=receptive_field, num_joints=model_num_joints, in_chans=2, embed_dim=model_embed_dim, joint_mapping=joint_mapping,
            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0, args=args, num_joints_out=num_joints_out)

    causal_shift = 0
    model_params = 0
    for parameter in model_pos.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    if torch.cuda.is_available():
        model_pos = nn.DataParallel(model_pos)
        model_pos = model_pos.cuda()
        model_pos_train = nn.DataParallel(model_pos_train)
        model_pos_train = model_pos_train.cuda()
    print(f'INFO: CUDA available: {torch.cuda.is_available()}', flush=True)

    is_fine_tuning = False
    if args.resume or args.evaluate:
        chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage, weights_only=False)
        state_dict = checkpoint['model_pos']
        
        head_weight_key = 'module.head.1.weight' if 'module.head.1.weight' in state_dict else 'head.1.weight'
        if head_weight_key in state_dict:
            if state_dict[head_weight_key].shape[0] != num_joints_out * 3:
                print('INFO: joint count mismatch ({} vs {}), removing head and mismatched layers for fine-tuning'.format(
                    state_dict[head_weight_key].shape[0] // 3, num_joints_out))
                keys_to_remove = [k for k in state_dict.keys() if 'head' in k]
                freq_weight_key = 'module.Freq_embedding.weight' if 'module.Freq_embedding.weight' in state_dict else 'Freq_embedding.weight'
                if freq_weight_key in state_dict and state_dict[freq_weight_key].shape[1] != model_num_joints * 2:
                    keys_to_remove += [k for k in state_dict.keys() if 'Freq_embedding' in k]
                for k in keys_to_remove:
                    del state_dict[k]
                epoch = 0 # Reset epoch for fine-tuning
                is_fine_tuning = True
                    
        model_pos_train.load_state_dict(state_dict, strict=False)
        model_pos.load_state_dict(state_dict, strict=False)

    test_generator = UnchunkedGenerator(None, poses_valid, poses_valid_2d,
                                        pad=pad, causal_shift=causal_shift, augment=False,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

    print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

    def evaluate(test_generator, action=None, return_predictions=False):
        epoch_loss_3d_pos = 0
        epoch_loss_3d_pos_procrustes = 0
        epoch_loss_3d_pos_scale = 0
        epoch_loss_3d_vel = 0
        with torch.no_grad():
            model_pos.eval()
            N = 0
            for cam, batch, batch_2d in test_generator.next_epoch():
                inputs_2d = torch.from_numpy(batch_2d.astype('float32')) 
                inputs_3d = torch.from_numpy(batch.astype('float32'))

                inputs_2d_flip = inputs_2d.clone()
                inputs_2d_flip [:, :, :, 0] *= -1
                inputs_2d_flip[:, :, kps_left + kps_right,:] = inputs_2d_flip[:, :, kps_right + kps_left,:]

                inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)
                inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d)

                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                    inputs_2d_flip = inputs_2d_flip.cuda()
                    inputs_3d = inputs_3d.cuda()
                inputs_3d[:, :, 0] = 0

                predicted_3d_pos = model_pos(inputs_2d)
                predicted_3d_pos_flip = model_pos(inputs_2d_flip)

                predicted_3d_pos_flip[:, :, :, 0] *= -1
                predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :, joints_right + joints_left]

                predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1, keepdim=True)

                if return_predictions:
                    return predicted_3d_pos.squeeze(0).cpu().numpy()

                error = mpjpe(predicted_3d_pos, inputs_3d)
                epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()
                epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
                N += inputs_3d.shape[0] * inputs_3d.shape[1]

                inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                predicted_3d_pos_np = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

                epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos_np, inputs)
                epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos_np, inputs)

        scale_factor = 100 if args.dataset == 'dhp19' else 1000
        e1 = (epoch_loss_3d_pos / N)*scale_factor
        e2 = (epoch_loss_3d_pos_procrustes / N)*scale_factor
        e3 = (epoch_loss_3d_pos_scale / N)*scale_factor
        ev = (epoch_loss_3d_vel / N)*scale_factor
        print('Protocol #1 Error (MPJPE):', e1, 'mm')
        print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
        print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
        print('Velocity Error (MPJVE):', ev, 'mm')
        return e1, e2, e3, ev

    if not args.evaluate:
        cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)
        print(f'INFO: Loaded {len(poses_train) if poses_train is not None else 0} train sequences', flush=True)
        optimizer = optim.AdamW(model_pos_train.parameters(), lr=args.learning_rate, weight_decay=0.1)
        lr = args.learning_rate
        lr_decay = args.lr_decay
        losses_3d_train = []
        losses_3d_valid = []
        epoch = 0

        train_generator = ChunkedGenerator(args.batch_size//args.stride, None, poses_train, poses_train_2d, args.stride,
                                           pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                           kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

        if args.resume and not is_fine_tuning:
            epoch = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
                train_generator.set_random_state(checkpoint['random_state'])
            lr = checkpoint['lr']

        while epoch < args.epochs:
            start_time = time()
            epoch_loss_3d_train = 0
            N = 0
            model_pos_train.train()
            print(f"Starting Epoch {epoch + 1}/{args.epochs}", flush=True)
            try:
                for _, batch_3d, batch_2d in tqdm(train_generator.next_epoch(), desc=f'Epoch {epoch + 1}/{args.epochs}', total=train_generator.num_batches, ncols=100, leave=False):
                    inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                    inputs_3d[:, :, 0] = 0
                    optimizer.zero_grad()
                    predicted_3d_pos = model_pos_train(inputs_2d)
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]
                    loss_3d_pos.backward()
                    optimizer.step()
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"ERROR in training loop: {e}", flush=True)
                traceback.print_exc()
                sys.exit(1)

            losses_3d_train.append(epoch_loss_3d_train / N)
            with torch.no_grad():
                model_pos.load_state_dict(model_pos_train.state_dict(), strict=False)
                model_pos.eval()
                if not args.no_eval:
                    e1, e2, e3, ev = evaluate(test_generator)
                    losses_3d_valid.append(e1 / (100 if args.dataset == 'dhp19' else 1000))
                else:
                    losses_3d_valid.append(0)

            elapsed = (time() - start_time) / 60
            scale_factor = 100 if args.dataset == 'dhp19' else 1000
            print('[%d] time %.2f lr %f 3d_train %f 3d_valid %f' % (epoch + 1, elapsed, lr, losses_3d_train[-1] * scale_factor, losses_3d_valid[-1] * scale_factor), flush=True)
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
            epoch += 1
            if epoch % args.checkpoint_frequency == 0:
                torch.save({'epoch': epoch, 'lr': lr, 'random_state': train_generator.random_state(), 'optimizer': optimizer.state_dict(), 'model_pos': model_pos_train.state_dict()}, os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch)))

    else:
        evaluate(test_generator)