# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random
import os
import sys
import errno
from datetime import datetime
from time import time
from einops import rearrange
from contextlib import nullcontext

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch.backends.cudnn import deterministic as cudnn_deterministic
from torch.backends.cudnn import benchmark as cudnn_benchmark
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from common.loss import (
    mpjpe,
    mpjpe_diffusion,
    mpjpe_diffusion_all_min,
    mpjpe_diffusion_reproj,
    p_mpjpe_diffusion,
    p_mpjpe_diffusion_all_min,
    p_mpjpe_diffusion_reproj,
)
from common.generators import ChunkedGenerator_Seq, UnchunkedGenerator_Seq
from common.utils import deterministic_random
from common.logging import (
    Logger,
    log_params_from_omegaconf_dict,
    log_metrics_to_mlflow,
    save_state
)
from common.camera import (
    normalize_screen_coordinates,
    project_to_2d,
    camera_to_world,
    image_coordinates
)
from common.diffusionpose import D3DP
from common.h3wb_dataset import Human3WBDataset
from common.utils import center_pose_at_root, center_pose_parts, wb_pose_from_parts


# >> Moved outside main routine <<
def fetch(
    subjects,
    keypoints,  # New
    dataset,  # New
    stride,  # New
    action_filter=None,
    subset=1,
    parse_3d_poses=True,
):
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
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    # >> Moved to function arguments <<
    # stride = args.experiment.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d


# >> Moved outside main routine <<
def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    assert inputs_2d.shape[:-1] == inputs_3d.shape[:-1], "2d and 3d inputs shape must be same! " + str(
        inputs_2d.shape) + str(inputs_3d.shape)
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = torch.squeeze(inputs_3d)

    if inputs_2d_p.shape[0] / receptive_field > inputs_2d_p.shape[0] // receptive_field:
        out_num = inputs_2d_p.shape[0] // receptive_field + 1
    elif inputs_2d_p.shape[0] / receptive_field == inputs_2d_p.shape[0] // receptive_field:
        out_num = inputs_2d_p.shape[0] // receptive_field

    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    eval_input_3d = torch.empty(out_num, receptive_field, inputs_3d_p.shape[1], inputs_3d_p.shape[2])

    for i in range(out_num - 1):
        eval_input_2d[i, :, :, :] = inputs_2d_p[i * receptive_field:i * receptive_field + receptive_field, :, :]
        eval_input_3d[i, :, :, :] = inputs_3d_p[i * receptive_field:i * receptive_field + receptive_field, :, :]
    if inputs_2d_p.shape[0] < receptive_field:
        from torch.nn import functional as F
        pad_right = receptive_field - inputs_2d_p.shape[0]
        inputs_2d_p = rearrange(inputs_2d_p, 'b f c -> f c b')
        inputs_2d_p = F.pad(inputs_2d_p, (0, pad_right), mode='replicate')
        # inputs_2d_p = np.pad(inputs_2d_p, ((0, receptive_field-inputs_2d_p.shape[0]), (0, 0), (0, 0)), 'edge')
        inputs_2d_p = rearrange(inputs_2d_p, 'f c b -> b f c')
    if inputs_3d_p.shape[0] < receptive_field:
        pad_right = receptive_field - inputs_3d_p.shape[0]
        inputs_3d_p = rearrange(inputs_3d_p, 'b f c -> f c b')
        inputs_3d_p = F.pad(inputs_3d_p, (0, pad_right), mode='replicate')
        inputs_3d_p = rearrange(inputs_3d_p, 'f c b -> b f c')
    eval_input_2d[-1, :, :, :] = inputs_2d_p[-receptive_field:, :, :]
    eval_input_3d[-1, :, :, :] = inputs_3d_p[-receptive_field:, :, :]

    return eval_input_2d, eval_input_3d


def decode_parts_to_trajectory(dataset, preds, gt_root_joints):
    preds_decode = preds.clone()
    body_kps = [0] + [ele + 1 for ele in dataset.metadata['body']] + [ele + 1 for ele in dataset.metadata['left_foot']] + [
        ele + 1 for ele in dataset.metadata['right_foot']]
    face_kps = [ele + 1 for ele in dataset.metadata['face']]
    hand_l_kps = [ele + 1 for ele in dataset.metadata['left_hand']]
    hand_r_kps = [ele + 1 for ele in dataset.metadata['right_hand']]
    # preds_decode[..., body_kps, :] += gt_root_joints[:, :, 1:2] - gt_root_joints[:, :, 0:1]
    if len(preds.shape) == 4:
        # Runs during training
        preds_decode[:, :, body_kps] += gt_root_joints[:, :, 0:1]
        preds_decode[:, :, face_kps] += gt_root_joints[:, :, 1:2]
        preds_decode[:, :, hand_l_kps] += gt_root_joints[:, :, 2:3]
        preds_decode[:, :, hand_r_kps] += gt_root_joints[:, :, 3:]
    elif len(preds.shape) == 5:
        # During Validation
        # if not gt_roots:
        #     preds_decode[:, :, :, body_kps] += gt_root_joints[:, :, :, 0:1]  # .unsqueeze(1)
        #     preds_decode[:, :, :, face_kps] += gt_root_joints[:, :, :, 1:2]  # .unsqueeze(1)
        #     preds_decode[:, :, :, hand_l_kps] += gt_root_joints[:, :, :, 2:3]  # .unsqueeze(1)
        #     preds_decode[:, :, :, hand_r_kps] += gt_root_joints[:, :, :, 3:]  # .unsqueeze(1)
        # else:
        preds_decode[:, :, :, body_kps] += gt_root_joints[:, :, 0:1].unsqueeze(1)
        preds_decode[:, :, :, face_kps] += gt_root_joints[:, :, 1:2].unsqueeze(1)
        preds_decode[:, :, :, hand_l_kps] += gt_root_joints[:, :, 2:3].unsqueeze(1)
        preds_decode[:, :, :, hand_r_kps] += gt_root_joints[:, :, 3:].unsqueeze(1)
    elif len(preds.shape) == 6:
        # During Testing!
        preds_decode[:, :, :, :, body_kps] += gt_root_joints[:, :, :, :, 0:1]
        preds_decode[:, :, :, :, face_kps] += gt_root_joints[:, :, :, :, 1:2]
        preds_decode[:, :, :, :, hand_l_kps] += gt_root_joints[:, :, :, :, 2:3]
        preds_decode[:, :, :, :, hand_r_kps] += gt_root_joints[:, :, :, :, 3:]

    return preds_decode


# >> Moved outside main routine <<
def evaluate(
        dataset,
        test_generator,
        model_pos,  # New
        args,       # New
        kps_left,   # New
        kps_right,  # New
        receptive_field,  # New
        action=None,
        return_predictions=False,
        use_trajectory_model=False,
        newmodel=None
):
    epoch_loss_3d_pos = torch.zeros(args.ft2d.sampling_timesteps).cuda()
    epoch_loss_3d_pos_h = torch.zeros(args.ft2d.sampling_timesteps).cuda()
    epoch_loss_3d_pos_h_pb = torch.zeros(args.ft2d.sampling_timesteps).cuda()
    epoch_loss_3d_pos_mean = torch.zeros(args.ft2d.sampling_timesteps).cuda()
    epoch_loss_3d_pos_select = torch.zeros(args.ft2d.sampling_timesteps).cuda()

    epoch_loss_3d_pos_h_pb_body = torch.zeros(args.ft2d.sampling_timesteps).cuda()
    epoch_loss_3d_pos_h_pb_face = torch.zeros(args.ft2d.sampling_timesteps).cuda()
    epoch_loss_3d_pos_h_pb_left_hand = torch.zeros(args.ft2d.sampling_timesteps).cuda()
    epoch_loss_3d_pos_h_pb_right_hand = torch.zeros(args.ft2d.sampling_timesteps).cuda()

    # p-agg for parts
    epoch_loss_3d_pos_agg_pb = torch.zeros(args.ft2d.sampling_timesteps).cuda()
    epoch_loss_3d_pos_agg_pb_body = torch.zeros(args.ft2d.sampling_timesteps).cuda()
    epoch_loss_3d_pos_agg_pb_face = torch.zeros(args.ft2d.sampling_timesteps).cuda()
    epoch_loss_3d_pos_agg_pb_left_hand = torch.zeros(args.ft2d.sampling_timesteps).cuda()
    epoch_loss_3d_pos_agg_pb_right_hand = torch.zeros(args.ft2d.sampling_timesteps).cuda()

    epoch_loss_3d_pos_p2 = torch.zeros(args.ft2d.sampling_timesteps)
    epoch_loss_3d_pos_h_p2 = torch.zeros(args.ft2d.sampling_timesteps)
    epoch_loss_3d_pos_mean_p2 = torch.zeros(args.ft2d.sampling_timesteps)
    epoch_loss_3d_pos_select_p2 = torch.zeros(args.ft2d.sampling_timesteps)

    with torch.no_grad():
        if newmodel is not None:
            print('Loading comparison model')
            model_eval = newmodel
            chk_file_path = '/mnt/data3/home/zjl/workspace/3dpose/PoseFormer/checkpoint/train_pf_00/epoch_60.bin'
            print('Loading evaluate checkpoint of comparison model', chk_file_path)
            checkpoint = torch.load(chk_file_path, map_location=lambda storage, loc: storage)
            model_eval.load_state_dict(checkpoint['model_pos'], strict=False)
            model_eval.eval()
        else:
            model_eval = model_pos
            if not use_trajectory_model:
                # load best checkpoint
                if args.general.evaluate == '':
                    chk_file_path = os.path.join(args.general.checkpoint, 'best_epoch.bin')
                    print('Loading best checkpoint', chk_file_path)
                elif args.general.evaluate != '':
                    chk_file_path = os.path.join(args.general.checkpoint, args.general.evaluate)
                    print('Loading evaluate checkpoint', chk_file_path)
                checkpoint = torch.load(chk_file_path, map_location=lambda storage, loc: storage)
                print('This model was trained for {} epochs'.format(checkpoint['epoch']))
                # model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)
                model_eval.load_state_dict(checkpoint['model_pos'])
                model_eval.eval()
        # else:
        # model_traj.eval()
        N = 0
        iteration = 0

        # num_batches = test_generator.batch_num()
        quickdebug = args.ft2d.debug
        for cam, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            cam = torch.from_numpy(cam.astype('float32'))

            ##### apply test-time-augmentation (following Videopose3d)
            # TODO: Duplicated code --> should be put in seperate function ideally
            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip[:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

            ##### convert size
            inputs_3d_p = inputs_3d
            if newmodel is not None:
                def eval_data_prepare_pf(receptive_field, inputs_2d, inputs_3d):
                    inputs_2d_p = torch.squeeze(inputs_2d)
                    inputs_3d_p = inputs_3d.permute(1, 0, 2, 3)
                    padding = int(receptive_field // 2)
                    inputs_2d_p = rearrange(inputs_2d_p, 'b f c -> f c b')
                    inputs_2d_p = F.pad(inputs_2d_p, (padding, padding), mode='replicate')
                    inputs_2d_p = rearrange(inputs_2d_p, 'f c b -> b f c')
                    out_num = inputs_2d_p.shape[0] - receptive_field + 1
                    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
                    for i in range(out_num):
                        eval_input_2d[i, :, :, :] = inputs_2d_p[i:i + receptive_field, :, :]
                    return eval_input_2d, inputs_3d_p

                inputs_2d, inputs_3d = eval_data_prepare_pf(81, inputs_2d, inputs_3d_p)
                inputs_2d_flip, _ = eval_data_prepare_pf(81, inputs_2d_flip, inputs_3d_p)
            else:
                inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d_p)
                inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d_p)

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_2d_flip = inputs_2d_flip.cuda()
                inputs_3d = inputs_3d.cuda()
                cam = cam.cuda()

            inputs_traj = inputs_3d[:, :, :1].clone()
            if not args.general.part_based_model:
                inputs_3d = center_pose_at_root(inputs_3d)
            else:
                inputs_3d = center_pose_parts(inputs_3d, dataset=dataset)

            bs = args.model.batch_size
            total_batch = (inputs_3d.shape[0] + bs - 1) // bs

            for batch_cnt in range(total_batch):

                if (batch_cnt + 1) * bs > inputs_3d.shape[0]:
                    inputs_2d_single = inputs_2d[batch_cnt * bs:]
                    inputs_2d_flip_single = inputs_2d_flip[batch_cnt * bs:]
                    inputs_3d_single = inputs_3d[batch_cnt * bs:]
                    inputs_traj_single = inputs_traj[batch_cnt * bs:]
                else:
                    inputs_2d_single = inputs_2d[batch_cnt * bs:(batch_cnt + 1) * bs]
                    inputs_2d_flip_single = inputs_2d_flip[batch_cnt * bs:(batch_cnt + 1) * bs]
                    inputs_3d_single = inputs_3d[batch_cnt * bs:(batch_cnt + 1) * bs]
                    inputs_traj_single = inputs_traj[batch_cnt * bs:(batch_cnt + 1) * bs]

                predicted_3d_pos_single = model_eval(inputs_2d_single, inputs_3d_single,
                                                     input_2d_flip=inputs_2d_flip_single)  # b, t, h, f, j, c

                b_sz, t_sz, h_sz, f_sz, j_sz, c_sz = predicted_3d_pos_single.shape

                if args.general.part_based_model:
                    predicted_3d_pos_single = wb_pose_from_parts(predicted_3d_pos_single, dataset=dataset)
                    inputs_3d_single = wb_pose_from_parts(inputs_3d_single, dataset=dataset)

                if return_predictions:
                    return predicted_3d_pos_single.squeeze().cpu().numpy()

                batch_multiplier = inputs_3d_single.shape[0] * inputs_3d_single.shape[1]
                # 2d reprojection
                inputs_traj_single_all = inputs_traj_single.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)
                predicted_3d_pos_abs_single = predicted_3d_pos_single + inputs_traj_single_all

                predicted_3d_pos_abs_single = predicted_3d_pos_abs_single.reshape(b_sz * t_sz * h_sz * f_sz, j_sz, c_sz)
                cam_single_all = cam.repeat(b_sz * t_sz * h_sz * f_sz, 1)
                reproject_2d = project_to_2d(predicted_3d_pos_abs_single, cam_single_all)
                reproject_2d = reproject_2d.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, 2)

                error = mpjpe_diffusion_all_min(predicted_3d_pos_single, inputs_3d_single)  # J-Best
                error_h, _ = mpjpe_diffusion(predicted_3d_pos_single, inputs_3d_single)  # P-Best
                error_mean = mpjpe_diffusion_all_min(predicted_3d_pos_single, inputs_3d_single, mean_pos=True)  # P-Agg
                error_reproj_select = mpjpe_diffusion_reproj(predicted_3d_pos_single, inputs_3d_single, reproject_2d,
                                                             inputs_2d_single)  # J-Agg
                error_h_pb, error_parts = mpjpe_diffusion(
                    predicted_3d_pos_single.clone(),
                    inputs_3d_single.clone(),
                    part_based=True,
                    dataset=dataset,
                )  # P-Best Part-Based

                error_agg_pb, error_agg_parts = mpjpe_diffusion_all_min(
                    predicted_3d_pos_single.clone(),
                    inputs_3d_single.clone(),
                    mean_pos=True,
                    part_based=True,
                    dataset=dataset
                )  # P-Best Part-Based

                epoch_loss_3d_pos += batch_multiplier * error.clone()
                epoch_loss_3d_pos_h += batch_multiplier * error_h.clone()
                epoch_loss_3d_pos_h_pb += batch_multiplier * error_h_pb.clone()
                epoch_loss_3d_pos_mean += batch_multiplier * error_mean.clone()
                epoch_loss_3d_pos_select += batch_multiplier * error_reproj_select.clone()

                epoch_loss_3d_pos_h_pb_body += batch_multiplier * error_parts['body'].clone()
                epoch_loss_3d_pos_h_pb_face += batch_multiplier * error_parts['face'].clone()
                epoch_loss_3d_pos_h_pb_left_hand += batch_multiplier * error_parts['left_hand'].clone()
                epoch_loss_3d_pos_h_pb_right_hand += batch_multiplier * error_parts['right_hand'].clone()

                epoch_loss_3d_pos_agg_pb += batch_multiplier * error_agg_pb.clone()
                epoch_loss_3d_pos_agg_pb_body += batch_multiplier * error_agg_parts['body'].clone()
                epoch_loss_3d_pos_agg_pb_face += batch_multiplier * error_agg_parts['face'].clone()
                epoch_loss_3d_pos_agg_pb_left_hand += batch_multiplier * error_agg_parts['left_hand'].clone()
                epoch_loss_3d_pos_agg_pb_right_hand += batch_multiplier * error_agg_parts['right_hand'].clone()

                if args.ft2d.p2:
                    error_p2 = p_mpjpe_diffusion_all_min(predicted_3d_pos_single, inputs_3d_single)
                    error_h_p2 = p_mpjpe_diffusion(predicted_3d_pos_single, inputs_3d_single)
                    error_mean_p2 = p_mpjpe_diffusion_all_min(predicted_3d_pos_single, inputs_3d_single, mean_pos=True)
                    error_reproj_select_p2 = p_mpjpe_diffusion_reproj(predicted_3d_pos_single, inputs_3d_single,
                                                                      reproject_2d, inputs_2d_single)

                    epoch_loss_3d_pos_p2 += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * torch.from_numpy(
                        error_p2)
                    epoch_loss_3d_pos_h_p2 += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * torch.from_numpy(
                        error_h_p2)
                    epoch_loss_3d_pos_mean_p2 += inputs_3d_single.shape[0] * inputs_3d_single.shape[
                        1] * torch.from_numpy(error_mean_p2)
                    epoch_loss_3d_pos_select_p2 += inputs_3d_single.shape[0] * inputs_3d_single.shape[
                        1] * torch.from_numpy(error_reproj_select_p2)

                N += batch_multiplier

                if quickdebug:
                    if N == batch_multiplier:
                        break
            if quickdebug:
                if N == batch_multiplier:
                    break

    log_path = os.path.join(args.general.checkpoint,
                            'h36m_test_log_H%d_K%d.txt' % (args.ft2d.num_proposals, args.ft2d.sampling_timesteps))
    f = open(log_path, mode='a')
    if action is None:
        print('----------')
    else:
        print('----' + action + '----')
        f.write('----' + action + '----\n')

    e1 = (epoch_loss_3d_pos / N) * 1000
    e1_h = (epoch_loss_3d_pos_h / N) * 1000
    e1_h_pb = (epoch_loss_3d_pos_h_pb / N) * 1000
    e1_mean = (epoch_loss_3d_pos_mean / N) * 1000
    e1_select = (epoch_loss_3d_pos_select / N) * 1000

    e1_h_pb_body = (epoch_loss_3d_pos_h_pb_body / N) * 1000
    e1_h_pb_face =(epoch_loss_3d_pos_h_pb_face / N) * 1000
    e1_h_pb_left_hand =(epoch_loss_3d_pos_h_pb_left_hand / N) * 1000
    e1_h_pb_right_hand = (epoch_loss_3d_pos_h_pb_right_hand / N) * 1000

    e1_agg_pb = (epoch_loss_3d_pos_agg_pb / N) * 1000
    e1_agg_pb_body = (epoch_loss_3d_pos_agg_pb_body / N) * 1000
    e1_agg_pb_face =(epoch_loss_3d_pos_agg_pb_face / N) * 1000
    e1_agg_pb_left_hand =(epoch_loss_3d_pos_agg_pb_left_hand / N) * 1000
    e1_agg_pb_right_hand = (epoch_loss_3d_pos_agg_pb_right_hand / N) * 1000

    if args.ft2d.p2:
        e2 = (epoch_loss_3d_pos_p2 / N) * 1000
        e2_h = (epoch_loss_3d_pos_h_p2 / N) * 1000
        e2_mean = (epoch_loss_3d_pos_mean_p2 / N) * 1000
        e2_select = (epoch_loss_3d_pos_select_p2 / N) * 1000

    print('Test time augmentation:', args.model.test_time_augmentation)
    for ii in range(e1.shape[0]):
        log = 'step %d : Protocol #1 Error (MPJPE) J_Best: %f mm' % (ii, e1[ii].item())
        print(log)
        f.write(log + '\n')
        log = 'step %d : Protocol #1 Error (MPJPE) P_Best: %f mm' % (ii, e1_h[ii].item())
        print(log)

        log = 'step %d : Protocol #1 Error (MPJPE) P_Agg: %f mm' % (ii, e1_mean[ii].item())
        print(log)
        f.write(log + '\n')
        log = 'step %d : Protocol #1 Error (MPJPE) J_Agg: %f mm' % (ii, e1_select[ii].item())
        print(log)
        f.write(log + '\n')

        log = '-----------------> Part-Based Evaluation <-----------------'
        print(log)
        f.write(log + '\n')

        f.write(log + '\n')
        log = 'step %d : Protocol #1 Error (MPJPE) P_Best Part-Based: %f mm' % (ii, e1_h_pb[ii].item())
        print(log)
        f.write(log + '\n')

        log = 'step %d : Protocol #1 Error (MPJPE) P_Best Part-Based BODY: %f mm' % (ii, e1_h_pb_body[ii].item())
        print(log)
        f.write(log + '\n')

        log = 'step %d : Protocol #1 Error (MPJPE) P_Best Part-Based FACE: %f mm' % (ii, e1_h_pb_face[ii].item())
        print(log)
        f.write(log + '\n')

        log = 'step %d : Protocol #1 Error (MPJPE) P_Best Part-Based HANDS: %f mm' % (ii, (e1_h_pb_right_hand[ii].item() + e1_h_pb_left_hand[ii].item())/2.)
        print(log)
        f.write(log + '\n')

        log = 'step %d : Protocol #1 Error (MPJPE) P_Best Part-Based LEFT HAND: %f mm' % (ii, e1_h_pb_left_hand[ii].item())
        print(log)
        f.write(log + '\n')

        log = 'step %d : Protocol #1 Error (MPJPE) P_Best Part-Based RIGHT HAND: %f mm' % (ii, e1_h_pb_right_hand[ii].item())
        print(log)
        f.write(log + '\n')

        log = '-----------------> Part-Based Evaluation Aggregation <-----------------'
        print(log)
        f.write(log + '\n')

        f.write(log + '\n')
        log = 'step %d : Protocol #1 Error (MPJPE) P_Agg Part-Based: %f mm' % (ii, e1_agg_pb[ii].item())
        print(log)
        f.write(log + '\n')

        log = 'step %d : Protocol #1 Error (MPJPE) P_Agg Part-Based BODY: %f mm' % (ii, e1_agg_pb_body[ii].item())
        print(log)
        f.write(log + '\n')

        log = 'step %d : Protocol #1 Error (MPJPE) P_Agg Part-Based FACE: %f mm' % (ii, e1_agg_pb_face[ii].item())
        print(log)
        f.write(log + '\n')

        log = 'step %d : Protocol #1 Error (MPJPE) P_Agg Part-Based HANDS: %f mm' % (ii, (e1_agg_pb_right_hand[ii].item() + e1_agg_pb_left_hand[ii].item())/2.)
        print(log)
        f.write(log + '\n')

        log = 'step %d : Protocol #1 Error (MPJPE) P_Agg Part-Based LEFT HAND: %f mm' % (ii, e1_agg_pb_left_hand[ii].item())
        print(log)
        f.write(log + '\n')

        log = 'step %d : Protocol #1 Error (MPJPE) P_Agg Part-Based RIGHT HAND: %f mm' % (ii, e1_agg_pb_right_hand[ii].item())
        print(log)
        f.write(log + '\n')

        if args.ft2d.p2:
            print('step %d : Protocol #2 Error (MPJPE) J_Best:' % ii, e2[ii].item(), 'mm')
            f.write('step %d : Protocol #2 Error (MPJPE) J_Best: %f mm\n' % (ii, e2[ii].item()))
            print('step %d : Protocol #2 Error (MPJPE) P_Best:' % ii, e2_h[ii].item(), 'mm')
            f.write('step %d : Protocol #2 Error (MPJPE) P_Best: %f mm\n' % (ii, e2_h[ii].item()))
            print('step %d : Protocol #2 Error (MPJPE) P_Agg:' % ii, e2_mean[ii].item(), 'mm')
            f.write('step %d : Protocol #2 Error (MPJPE) P_Agg: %f mm\n' % (ii, e2_mean[ii].item()))
            print('step %d : Protocol #2 Error (MPJPE) J_Agg:' % ii, e2_select[ii].item(), 'mm')
            f.write('step %d : Protocol #2 Error (MPJPE) J_Agg: %f mm\n' % (ii, e2_select[ii].item()))

    print('----------')
    f.write('----------\n')

    f.close()

    if args.ft2d.p2:
        return e1, e1_h, e1_h_pb, e1_mean, e1_select, e2, e2_h, e2_mean, e2_select
    else:
        return e1, e1_h, e1_h_pb, e1_mean, e1_select, e1_h_pb_body, e1_h_pb_face, e1_h_pb_left_hand, e1_h_pb_right_hand, e1_agg_pb, e1_agg_pb_body, e1_agg_pb_face, e1_agg_pb_left_hand, e1_agg_pb_right_hand


# >> Moved outside main routine <<
def fetch_actions(actions, keypoints, dataset, stride):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []

    for subject, action in actions:
        poses_2d = keypoints[subject][action]
        for i in range(len(poses_2d)):  # Iterate across cameras
            out_poses_2d.append(poses_2d[i])

        poses_3d = dataset[subject][action]['positions_3d']
        assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
        for i in range(len(poses_3d)):  # Iterate across cameras
            out_poses_3d.append(poses_3d[i])

        if subject in dataset.cameras():
            cams = dataset.cameras()[subject]
            assert len(cams) == len(poses_2d), 'Camera count mismatch'
            for cam in cams:
                if 'intrinsic' in cam:
                    out_camera_params.append(cam['intrinsic'])

    # >> Moved to function arguments <<
    # stride = args.experiment.downsample
    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig):
    if args.general.evaluate != '':
        description = "Evaluate!"
    elif args.general.evaluate == '':

        description = "Train!"

    # initial setting
    TIMESTAMP = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())
    # tensorboard
    if not args.general.nolog:
        writer = SummaryWriter(args.general.log + '_' + TIMESTAMP)
        writer.add_text('description', description)
        writer.add_text('command', 'python ' + ' '.join(sys.argv))
        # logging setting
        logfile = os.path.join(args.general.log + '_' + TIMESTAMP, 'logging.log')
        sys.stdout = Logger(logfile)
    print(description)
    print("CUDA Device Count: ", torch.cuda.device_count())
    print("==> Using settings:")
    print(OmegaConf.to_yaml(args))

    # TODO: isolate in own function
    manualSeed = 1  # TODO: Should be in the config
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    cudnn_deterministic = True
    cudnn_benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.experiment.gpu

    # if not assign checkpoint path, Save checkpoint file into log folder
    if args.general.checkpoint == '':
        args.general.checkpoint = args.general.log + '_' + TIMESTAMP
    try:
        # Create checkpoint directory if it does not exist
        os.makedirs(args.general.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', args.general.checkpoint)

    # dataset loading
    # TODO: isolate in own function
    print('Loading dataset...')
    dataset_path = 'data/train_' + args.data.dataset + '.npz'

    dataset = Human3WBDataset(dataset_path)

    print('Preparing 3D data...')
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            if 'positions' in anim:
                positions_3d = []
                for ind, cam in enumerate(anim['cameras']):
                    pos_3d = anim['positions_3d'][ind]
                    pos_3d = pos_3d / 1000.  # lets divide by 1000 to convert meters
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

    print('Preparing 2D detections...')
    keypoints_metadata = dataset.keypoints_metadata
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())

    ################### 2D data preparation
    keypoints = {}
    for subject in dataset.subjects():
        keypoints[subject] = {}
        for action in dataset[subject].keys():
            keypoints[subject][action] = []
            for cam_idx, kps in enumerate(dataset[subject][action]['pose_2d']):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action].append(kps)

    subjects_train = args.data.subjects_train.split(',')
    # >> UNUSED <<
    # subjects_semi = [] if not args.data.subjects_unlabeled else args.data.subjects_unlabeled.split(',')
    if not args.general.render:
        subjects_test = args.data.subjects_test.split(',')
    else:
        subjects_test = [args.viz.viz_subject]

    action_filter = None if args.data.actions == '*' else args.data.actions.split(',')
    if action_filter is not None:
        print('Selected actions:', action_filter)

    # >> Modification <<
    cameras_valid, poses_valid, poses_valid_2d = fetch(
        subjects=subjects_test,
        keypoints=keypoints,
        dataset=dataset,
        stride=args.experiment.downsample,
        action_filter=action_filter,
    )

    # TODO: isolate model creation in own function
    # set receptive_field as number assigned
    receptive_field = args.model.number_of_frames
    print('INFO: Receptive field: {} frames'.format(receptive_field))
    if not args.general.nolog:
        writer.add_text(args.general.log + '_' + TIMESTAMP + '/Receptive field', str(receptive_field))
    pad = (receptive_field - 1) // 2  # Padding on each side
    min_loss = args.model.min_loss
    # >> UNUSED <<
    # width = cam['res_w']
    # height = cam['res_h']
    # num_joints = keypoints_metadata['num_joints']

    print('INFO: Creating the models')
    model_pos_train = D3DP(args, joints_left, joints_right, dataset=dataset, is_train=True)
    model_pos_test_temp = D3DP(args, joints_left, joints_right, dataset=dataset, is_train=False)
    model_pos = D3DP(args, joints_left, joints_right, dataset=dataset, is_train=False,
                     num_proposals=args.ft2d.num_proposals, sampling_timesteps=args.ft2d.sampling_timesteps)

    causal_shift = 0
    model_params = 0
    for parameter in model_pos.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params / 1000000, 'Million')
    if not args.general.nolog:
        writer.add_text(args.general.log + '_' + TIMESTAMP + '/Trainable parameter count', str(model_params / 1000000) + ' Million')

    # make model parallel
    if torch.cuda.is_available():
        model_pos = nn.DataParallel(model_pos)
        model_pos = model_pos.cuda()
        model_pos_train = nn.DataParallel(model_pos_train)
        model_pos_train = model_pos_train.cuda()
        model_pos_test_temp = nn.DataParallel(model_pos_test_temp)
        model_pos_test_temp = model_pos_test_temp.cuda()

    if args.general.resume or args.general.evaluate:
        chk_filename = os.path.join(args.general.checkpoint, args.general.resume if args.general.resume else args.general.evaluate)
        # chk_filename = args.general.resume or args.general.evaluate
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        print('This model was trained for {} epochs'.format(checkpoint['epoch']))
        model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)
        model_pos.load_state_dict(checkpoint['model_pos'], strict=False)

    test_generator = UnchunkedGenerator_Seq(cameras_valid, poses_valid, poses_valid_2d,
                                            pad=pad, causal_shift=causal_shift, augment=False,
                                            kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                            joints_right=joints_right)
    print('INFO: Testing on {} frames'.format(test_generator.num_frames()))
    if not args.general.nolog:
        writer.add_text(args.general.log + '_' + TIMESTAMP + '/Testing Frames', str(test_generator.num_frames()))

    if args.model.weighted_loss:
        weight = [1, 1, 1, 1, 1, 1, 1.5, 1.5, 4, 4, 4, 4, 1, 1, 2.5, 2.5, 2.5, 2.5] # these weights are from MixSTE
        weight.extend(116 * [1.0])
        w_mpjpe = torch.tensor(weight).cuda()
    else:
        w_mpjpe=None
    ###################
    # Training start
    mlflow_on = args.mlflow.mlflow_on
    if mlflow_on:
        # Lazy import of MLFlow if requested
        import mlflow as mlf
        mlf.set_tracking_uri(f"file://{args.mlflow.mlflow_uri}/mlruns")
        # TODO: replace args.checkpoint by better name
        mlf.set_experiment(args.mlflow.experiment)

    # Used to log to MLFlow or not depending on config

    context = mlf.start_run if mlflow_on else nullcontext
    with context():
        log_params_from_omegaconf_dict(args, mlflow_on=mlflow_on)

        # to facilitate retrival of exp data
        # TODO: Add experiment directory to logs
        # log_param_to_mlf("mlflow.experiment_dir", output_dur)

        if not args.general.evaluate:
            cameras_train, poses_train, poses_train_2d = fetch(
                subjects=subjects_train,
                keypoints=keypoints,
                dataset=dataset,
                stride=args.experiment.downsample,
                action_filter=action_filter,
                subset=args.experiment.subset,
            )

            lr = args.model.learning_rate
            optimizer = optim.AdamW(model_pos_train.parameters(), lr=lr, weight_decay=0.1)

            lr_decay = args.model.lr_decay
            losses_3d_train = []
            losses_3d_pos_train = []
            losses_3d_diff_train = []  # << XXX UNUSED --> DELETE?
            losses_3d_train_eval = []
            losses_3d_valid = []
            losses_pb_3d_valid = []
            # losses_3d_depth_valid = []  << XXX UNUSED

            epoch = 0
            # XXX >> UNUSED <<
            # best_epoch = 0
            # initial_momentum = 0.1
            # final_momentum = 0.001

            # get training data
            # TODO: replaced stride by number of frames here for HP search --> should be changed later
            # train_generator = ChunkedGenerator_Seq(args.model.batch_size // args.model.stride, cameras_train, poses_train, poses_train_2d,
            train_generator = ChunkedGenerator_Seq(args.model.batch_size // args.model.number_of_frames, cameras_train, poses_train, poses_train_2d,
                                                args.model.number_of_frames,
                                                pad=pad, causal_shift=causal_shift, shuffle=True,
                                                augment=args.model.data_augmentation,
                                                kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                                joints_right=joints_right)
            train_generator_eval = UnchunkedGenerator_Seq(cameras_train, poses_train, poses_train_2d,
                                                        pad=pad, causal_shift=causal_shift, augment=False)
            print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))
            if not args.general.nolog:
                writer.add_text(args.general.log + '_' + TIMESTAMP + '/Training Frames', str(train_generator_eval.num_frames()))

            if args.general.resume:
                epoch = checkpoint['epoch']
                if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    train_generator.set_random_state(checkpoint['random_state'])
                else:
                    print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
                if not args.model.coverlr:
                    lr = checkpoint['lr']

            print('** Note: reported losses are averaged over all frames.')
            print('** The final evaluation will be carried out after the last training epoch.')

            # Pos model only
            while epoch < args.model.epochs:
                start_time = time()
                epoch_loss_3d_train = 0
                epoch_loss_3d_pos_train = 0
                # XXX >> UNUSED <<
                # epoch_loss_3d_diff_train = 0
                # epoch_loss_traj_train = 0
                # epoch_loss_2d_train_unlabeled = 0
                N = 0
                # N_semi = 0  << XXX UNUSED
                model_pos_train.train()
                iteration = 0

                num_batches = train_generator.batch_num()

                # Just train 1 time, for quick debug
                quickdebug = args.ft2d.debug
                for cameras_train, batch_3d, batch_2d in train_generator.next_epoch():

                    if iteration % 10 == 0:
                        print("%d/%d" % (iteration, num_batches))

                    if cameras_train is not None:
                        cameras_train = torch.from_numpy(cameras_train.astype('float32'))
                    inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                        if cameras_train is not None:
                            cameras_train = cameras_train.cuda()

                    if not args.general.part_based_model:
                        inputs_3d = center_pose_at_root(inputs_3d)
                    else:
                        inputs_3d = center_pose_parts(
                            inputs_3d, dataset=dataset
                        )

                    optimizer.zero_grad()

                    # Predict 3D poses
                    predicted_3d_pos = model_pos_train(inputs_2d, inputs_3d)

                    '''Optimize over wholebody'''
                    if args.general.part_based_model and args.model.wb_loss:
                        # NOTE: moved below lines from diff model to here since they are post-processing steps.
                        predicted_3d_pos = wb_pose_from_parts(predicted_3d_pos, dataset=dataset)
                        inputs_3d = wb_pose_from_parts(inputs_3d, dataset=dataset)

                    # TODO: This is usually MSE in DDPM! Ablate this, as it may be better for 3D-HPE, as in supervised setting
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d, weights=w_mpjpe, mse_loss=args.model.mse_loss)
                    # loss_3d_pos = torch.mean((predicted_3d_pos - inputs_3d)**2)

                    loss_total = loss_3d_pos

                    # >>> MODIFIED <<<
                    loss_total.backward()

                    epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_total.item()
                    epoch_loss_3d_pos_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    optimizer.step()
                    # del inputs_3d, loss_3d_pos, predicted_3d_pos
                    # torch.cuda.empty_cache()

                    iteration += 1

                    if quickdebug:
                        if N == inputs_3d.shape[0] * inputs_3d.shape[1]:
                            break

                    # if iteration == 3:
                    #     break

                epoch_loss_3d_train /= N
                epoch_loss_3d_train_mm = epoch_loss_3d_train * 1000
                losses_3d_train.append(epoch_loss_3d_train)
                epoch_loss_3d_pos_train /= N
                epoch_loss_3d_pos_train_mm = epoch_loss_3d_pos_train * 1000
                losses_3d_pos_train.append(epoch_loss_3d_pos_train)

                # Store training loss for logging
                metrics_to_log = {
                    "tr_loss": epoch_loss_3d_pos_train_mm,
                }

                # XXX: DELETE?
                # torch.cuda.empty_cache()

                # End-of-epoch evaluation
                with torch.no_grad():
                    model_pos_test_temp.load_state_dict(model_pos_train.state_dict(), strict=False)
                    model_pos_test_temp.eval()

                    epoch_loss_3d_valid = None
                    epoch_pbloss_3d_valid = None
                    # XXX >> UNUSED <<
                    # epoch_loss_3d_depth_valid = 0
                    # epoch_loss_traj_valid = 0
                    # epoch_loss_2d_valid = 0
                    # epoch_loss_3d_vel = 0
                    N = 0
                    iteration = 0
                    if not args.experiment.no_eval:
                        # Evaluate on test set
                        for cam, batch, batch_2d in test_generator.next_epoch():
                            inputs_3d = torch.from_numpy(batch.astype('float32'))
                            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

                            ##### apply test-time-augmentation (following Videopose3d)
                            inputs_2d_flip = inputs_2d.clone()
                            inputs_2d_flip[:, :, :, 0] *= -1
                            inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

                            ##### convert size
                            inputs_3d_p = inputs_3d
                            inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d_p)
                            inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d_p)

                            if torch.cuda.is_available():
                                inputs_3d = inputs_3d.cuda()
                                inputs_2d = inputs_2d.cuda()
                                inputs_2d_flip = inputs_2d_flip.cuda()

                            if not args.general.part_based_model:
                                inputs_3d = center_pose_at_root(inputs_3d)
                            else:
                                inputs_3d = center_pose_parts(
                                    inputs_3d, dataset=dataset,
                                )

                            predicted_3d_pos = model_pos_test_temp(inputs_2d, inputs_3d,
                                                                input_2d_flip=inputs_2d_flip)  # input_class_embeddings, deform_index=0 - b, t, h, f, j, c

                            if args.general.part_based_model:
                                # NOTE: moved below lines from diff model to here since they are post-processing steps.
                                predicted_3d_pos = wb_pose_from_parts(predicted_3d_pos, dataset=dataset)
                                inputs_3d = wb_pose_from_parts(inputs_3d, dataset=dataset)

                            error, _ = mpjpe_diffusion(predicted_3d_pos, inputs_3d)
                            part_based_error, error_parts = mpjpe_diffusion(
                                predicted_3d_pos, inputs_3d,
                                part_based=True,
                                dataset=dataset,
                            )

                            if iteration == 0:
                                epoch_loss_3d_valid = inputs_3d.shape[0] * inputs_3d.shape[1] * error.clone()
                                epoch_pbloss_3d_valid = inputs_3d.shape[0] * inputs_3d.shape[1] * part_based_error.clone()
                            else:
                                epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * error.clone()
                                epoch_pbloss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * part_based_error.clone()

                            N += inputs_3d.shape[0] * inputs_3d.shape[1]

                            iteration += 1

                            if quickdebug:
                                if N == inputs_3d.shape[0] * inputs_3d.shape[1]:
                                    break

                        epoch_loss_3d_valid /= N
                        epoch_loss_3d_valid_mm = epoch_loss_3d_valid[0] * 1000
                        losses_3d_valid.append(epoch_loss_3d_valid)
                        epoch_pbloss_3d_valid /= N
                        epoch_pbloss_3d_valid_mm = epoch_pbloss_3d_valid[0] * 1000
                        losses_pb_3d_valid.append(epoch_pbloss_3d_valid)
                        metrics_to_log["val_pb_mpjpe"] = epoch_pbloss_3d_valid_mm
                        metrics_to_log["val_mpjpe"] = epoch_loss_3d_valid_mm

                elapsed = (time() - start_time) / 60

                if args.experiment.no_eval:
                    log = '[%d] time %.2f lr %f 3d_train %f 3d_pos_train %f 3d_diff_train %f' % (
                        epoch + 1,
                        elapsed,
                        lr,
                        epoch_loss_3d_train_mm,
                        epoch_loss_3d_pos_train_mm,
                        losses_3d_diff_train[-1] * 1000
                    )
                else:
                    log = '[%d] time %.2f lr %f 3d_train %f 3d_pos_train %f 3d_pos_valid %f 3d_pb_pos_valid %f' % (
                        epoch + 1,
                        elapsed,
                        lr,
                        epoch_loss_3d_train_mm,
                        epoch_loss_3d_pos_train_mm,
                        epoch_loss_3d_valid_mm,
                        epoch_pbloss_3d_valid_mm,
                    )

                    if not args.general.nolog:
                        # writer.add_scalar("Loss/3d training eval loss", losses_3d_train_eval[-1] * 1000, epoch+1)
                        writer.add_scalar("Loss/3d validation loss", losses_3d_valid[-1] * 1000, epoch + 1)

                print(log)

                log_path = os.path.join(args.general.checkpoint, 'training_log.txt')
                f = open(log_path, mode='a')
                f.write(log + '\n')
                f.close()

                if not args.general.nolog:
                    writer.add_scalar("Loss/3d training loss", epoch_loss_3d_train_mm, epoch + 1)
                    writer.add_scalar("Parameters/learing rate", lr, epoch + 1)
                    writer.add_scalar('Parameters/training time per epoch', elapsed, epoch + 1)
                # Decay learning rate exponentially
                lr *= lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay
                epoch += 1

                # Decay BatchNorm momentum
                # momentum = initial_momentum * np.exp(-epoch/args.model.epochs * np.log(initial_momentum/final_momentum))
                # model_pos_train.set_bn_momentum(momentum)

                # Save checkpoint if necessary
                if epoch % args.general.checkpoint_frequency == 0:
                    save_state(
                        model=model_pos_train,
                        optimizer=optimizer,
                        scheduler=None,  # XXX Should we add a scheduler?
                        epoch_no=epoch,
                        lr=lr,
                        foldername=args.general.checkpoint,
                        log_in_mlf=mlflow_on,
                    )

                #### save best checkpoint
                if epoch_loss_3d_valid_mm < min_loss:
                    min_loss = epoch_loss_3d_valid_mm
                    # best_epoch = epoch  << XXX UNUSED
                    save_state(
                        model=model_pos_train,
                        optimizer=optimizer,
                        scheduler=None,  # XXX Should we add a scheduler?
                        random_state=train_generator.random_state(),
                        epoch_no=epoch,
                        lr=lr,
                        foldername=args.general.checkpoint,
                        log_in_mlf=mlflow_on,
                        tag="best_epoch"
                    )

                    f = open(log_path, mode='a')
                    f.write('best epoch\n')
                    f.close()

                    # Log when validation loss improves
                    metrics_to_log.update(
                        {
                            "best_epoch_loss": epoch,
                            "best_val_loss": epoch_loss_3d_valid_mm,
                        }
                    )

                log_metrics_to_mlflow(
                    metrics=metrics_to_log,
                    step=epoch,
                    mlflow_on=mlflow_on,
                )

                # Save training curves after every epoch, as .png images (if requested)
                if args.general.export_training_curves and epoch > 3:
                    if 'matplotlib' not in sys.modules:
                        import matplotlib

                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt

                    plt.figure()
                    epoch_x = np.arange(3, len(losses_3d_train)) + 1
                    plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
                    plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
                    plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
                    plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
                    plt.ylabel('MPJPE (m)')
                    plt.xlabel('Epoch')
                    plt.xlim((3, epoch))
                    fig_name = os.path.join(args.general.checkpoint, 'loss_3d.png')
                    plt.savefig(fig_name)
                    if mlflow_on:
                        mlf.log_artifact(fig_name)

                    plt.close('all')
        # Training end

        # Evaluate
        print('Evaluating...')
        all_actions = {}
        all_actions_flatten = []
        all_actions_by_subject = {}
        for subject in subjects_test:
            if subject not in all_actions_by_subject:
                all_actions_by_subject[subject] = {}

            for action in dataset[subject].keys():
                action_name = action.split(' ')[0]
                if action_name not in all_actions:
                    all_actions[action_name] = []
                if action_name not in all_actions_by_subject[subject]:
                    all_actions_by_subject[subject][action_name] = []
                all_actions[action_name].append((subject, action))
                all_actions_flatten.append((subject, action))
                all_actions_by_subject[subject][action_name].append((subject, action))

        def run_evaluation(actions, action_filter=None):
            errors_p1 = []
            errors_p1_h = []
            errors_p1_h_pb = []
            errors_p1_mean = []
            errors_p1_select = []
            errors_p1_h_pb_body = []
            errors_p1_h_pb_face = []
            errors_p1_h_pb_left_hand = []
            errors_p1_h_pb_right_hand = []

            # p-agg
            errors_p1_agg_pb = []
            errors_p1_agg_pb_body = []
            errors_p1_agg_pb_face = []
            errors_p1_agg_pb_left_hand = []
            errors_p1_agg_pb_right_hand = []

            errors_p2 = []
            errors_p2_h = []
            errors_p2_mean = []
            errors_p2_select = []

            for action_key in actions.keys():
                if action_filter is not None:
                    found = False
                    for a in action_filter:
                        if action_key.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                # >> Modification <<
                cameras_act, poses_act, poses_2d_act = fetch_actions(
                    actions=actions[action_key],
                    keypoints=keypoints,
                    dataset=dataset,
                    stride=args.experiment.downsample,
                )
                gen = UnchunkedGenerator_Seq(cameras_act, poses_act, poses_2d_act,
                                            pad=pad, causal_shift=causal_shift, augment=args.model.test_time_augmentation,
                                            kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                            joints_right=joints_right)

                #  >> Modification <<
                if args.ft2d.p2:
                    e1, e1_h, e1_h_pb, e1_mean, e1_select, e2, e2_h, e2_mean, e2_select = evaluate(
                        dataset,
                        test_generator=gen,
                        model_pos=model_pos,
                        args=args,
                        kps_left=kps_left,
                        kps_right=kps_right,
                        receptive_field=receptive_field,
                        action=action_key,
                    )
                else:
                    e1, e1_h, e1_h_pb, e1_mean, e1_select, e1_h_pb_body, e1_h_pb_face, e1_h_pb_left_hand, e1_h_pb_right_hand, e1_agg_pb, e1_agg_pb_body, e1_agg_pb_face, e1_agg_pb_left_hand, e1_agg_pb_right_hand = evaluate(
                        dataset,
                        test_generator=gen,
                        model_pos=model_pos,
                        args=args,
                        kps_left=kps_left,
                        kps_right=kps_right,
                        receptive_field=receptive_field,
                        action=action_key,
                    )

                errors_p1.append(e1)
                errors_p1_h.append(e1_h)
                errors_p1_h_pb.append(e1_h_pb)
                errors_p1_mean.append(e1_mean)
                errors_p1_select.append(e1_select)
                errors_p1_h_pb_body.append(e1_h_pb_body)
                errors_p1_h_pb_face.append(e1_h_pb_face)
                errors_p1_h_pb_left_hand.append(e1_h_pb_left_hand)
                errors_p1_h_pb_right_hand.append(e1_h_pb_right_hand)

                errors_p1_agg_pb.append(e1_agg_pb)
                errors_p1_agg_pb_body.append(e1_agg_pb_body)
                errors_p1_agg_pb_face.append(e1_agg_pb_face)
                errors_p1_agg_pb_left_hand.append(e1_agg_pb_left_hand)
                errors_p1_agg_pb_right_hand.append(e1_agg_pb_right_hand)

                if args.ft2d.p2:
                    errors_p2.append(e2)
                    errors_p2_h.append(e2_h)
                    errors_p2_mean.append(e2_mean)
                    errors_p2_select.append(e2_select)

            errors_p1 = torch.stack(errors_p1)
            errors_p1_actionwise = torch.mean(errors_p1, dim=0)
            errors_p1_h = torch.stack(errors_p1_h)
            errors_p1_actionwise_h = torch.mean(errors_p1_h, dim=0)
            errors_p1_h_pb = torch.stack(errors_p1_h_pb)
            errors_p1_actionwise_h_pb = torch.mean(errors_p1_h_pb, dim=0)
            errors_p1_mean = torch.stack(errors_p1_mean)
            errors_p1_actionwise_mean = torch.mean(errors_p1_mean, dim=0)
            errors_p1_select = torch.stack(errors_p1_select)
            errors_p1_actionwise_select = torch.mean(errors_p1_select, dim=0)
            # parts
            errors_p1_h_pb_body = torch.stack(errors_p1_h_pb_body)
            errors_p1_actionwise_h_pb_body = torch.mean(errors_p1_h_pb_body, dim=0)
            errors_p1_h_pb_face = torch.stack(errors_p1_h_pb_face)
            errors_p1_actionwise_h_pb_face = torch.mean(errors_p1_h_pb_face, dim=0)
            errors_p1_h_pb_left_hand = torch.stack(errors_p1_h_pb_left_hand)
            errors_p1_actionwise_h_pb_left_hand = torch.mean(errors_p1_h_pb_left_hand, dim=0)
            errors_p1_h_pb_right_hand = torch.stack(errors_p1_h_pb_right_hand)
            errors_p1_actionwise_h_pb_right_hand = torch.mean(errors_p1_h_pb_right_hand, dim=0)

            # parts- agg
            errors_p1_agg_pb = torch.stack(errors_p1_agg_pb)
            errors_p1_actionwise_agg_pb = torch.mean(errors_p1_agg_pb, dim=0)
            errors_p1_agg_pb_body = torch.stack(errors_p1_agg_pb_body)
            errors_p1_actionwise_agg_pb_body = torch.mean(errors_p1_agg_pb_body, dim=0)
            errors_p1_agg_pb_face = torch.stack(errors_p1_agg_pb_face)
            errors_p1_actionwise_agg_pb_face = torch.mean(errors_p1_agg_pb_face, dim=0)
            errors_p1_agg_pb_left_hand = torch.stack(errors_p1_agg_pb_left_hand)
            errors_p1_actionwise_agg_pb_left_hand = torch.mean(errors_p1_agg_pb_left_hand, dim=0)
            errors_p1_agg_pb_right_hand = torch.stack(errors_p1_agg_pb_right_hand)
            errors_p1_actionwise_agg_pb_right_hand = torch.mean(errors_p1_agg_pb_right_hand, dim=0)

            if args.ft2d.p2:
                errors_p2 = torch.stack(errors_p2)
                errors_p2_actionwise = torch.mean(errors_p2, dim=0)
                errors_p2_h = torch.stack(errors_p2_h)
                errors_p2_actionwise_h = torch.mean(errors_p2_h, dim=0)
                errors_p2_mean = torch.stack(errors_p2_mean)
                errors_p2_actionwise_mean = torch.mean(errors_p2_mean, dim=0)
                errors_p2_select = torch.stack(errors_p2_select)
                errors_p2_actionwise_select = torch.mean(errors_p2_select, dim=0)

            log_path = os.path.join(args.general.checkpoint,
                                    'h36m_test_log_H%d_K%d.txt' % (args.ft2d.num_proposals, args.ft2d.sampling_timesteps))
            f = open(log_path, mode='a')
            for ii in range(errors_p1_actionwise.shape[0]):
                log = 'step %d Protocol #1   (MPJPE) action-wise average J_Best: %f mm' % (
                ii, errors_p1_actionwise[ii].item())
                print(log)
                f.write(log + '\n')
                log = 'step %d Protocol #1   (MPJPE) action-wise average P_Best: %f mm' % (
                ii, errors_p1_actionwise_h[ii].item())
                print(log)
                f.write(log + '\n')

                log = 'step %d Protocol #1   (MPJPE) action-wise average P_Agg: %f mm' % (
                ii, errors_p1_actionwise_mean[ii].item())
                print(log)
                f.write(log + '\n')
                log = 'step %d Protocol #1   (MPJPE) action-wise average J_Agg: %f mm' % (
                    ii, errors_p1_actionwise_select[ii].item())
                print(log)
                f.write(log + '\n')

                # Part-based Evaluation
                log = '-----------------> Part-Based Evaluation <-----------------'
                print(log)
                f.write(log + '\n')

                log = 'step %d Protocol #1   (MPJPE) action-wise average P_Best (Part-Based): %f mm' % (
                ii, errors_p1_actionwise_h_pb[ii].item())
                print(log)
                f.write(log + '\n')

                log = 'step %d Protocol #1   (MPJPE) action-wise average P_Best (Part-Based) BODY: %f mm' % (
                    ii, errors_p1_actionwise_h_pb_body[ii].item())
                print(log)
                f.write(log + '\n')

                log = 'step %d Protocol #1   (MPJPE) action-wise average P_Best (Part-Based) FACE: %f mm' % (
                    ii, errors_p1_actionwise_h_pb_face[ii].item())
                print(log)
                f.write(log + '\n')

                log = 'step %d Protocol #1   (MPJPE) action-wise average P_Best (Part-Based) HANDS: %f mm' % (
                    ii, (errors_p1_actionwise_h_pb_left_hand[ii].item() + errors_p1_actionwise_h_pb_right_hand[ii].item())/2.)
                print(log)
                f.write(log + '\n')

                log = 'step %d Protocol #1   (MPJPE) action-wise average P_Best (Part-Based) LEFT HAND: %f mm' % (
                    ii, errors_p1_actionwise_h_pb_left_hand[ii].item())
                print(log)
                f.write(log + '\n')

                log = 'step %d Protocol #1   (MPJPE) action-wise average P_Best (Part-Based) RIGHT HAND: %f mm' % (
                    ii, errors_p1_actionwise_h_pb_right_hand[ii].item())
                print(log)
                f.write(log + '\n')

                # Part-based AGG Evaluation
                log = '-----------------> Part-Based Agg Evaluation <-----------------'
                print(log)
                f.write(log + '\n')

                log = 'step %d Protocol #1   (MPJPE) action-wise average P_Agg (Part-Based): %f mm' % (
                ii, errors_p1_actionwise_agg_pb[ii].item())
                print(log)
                f.write(log + '\n')

                log = 'step %d Protocol #1   (MPJPE) action-wise average P_Agg (Part-Based) BODY: %f mm' % (
                    ii, errors_p1_actionwise_agg_pb_body[ii].item())
                print(log)
                f.write(log + '\n')

                log = 'step %d Protocol #1   (MPJPE) action-wise average P_Agg (Part-Based) FACE: %f mm' % (
                    ii, errors_p1_actionwise_agg_pb_face[ii].item())
                print(log)
                f.write(log + '\n')

                log = 'step %d Protocol #1   (MPJPE) action-wise average P_Agg (Part-Based) HANDS: %f mm' % (
                    ii, (errors_p1_actionwise_agg_pb_left_hand[ii].item() + errors_p1_actionwise_agg_pb_right_hand[ii].item())/2.)
                print(log)
                f.write(log + '\n')

                log = 'step %d Protocol #1   (MPJPE) action-wise average P_Agg (Part-Based) LEFT HAND: %f mm' % (
                    ii, errors_p1_actionwise_agg_pb_left_hand[ii].item())
                print(log)
                f.write(log + '\n')

                log = 'step %d Protocol #1   (MPJPE) action-wise average P_Agg (Part-Based) RIGHT HAND: %f mm' % (
                    ii, errors_p1_actionwise_agg_pb_right_hand[ii].item())
                print(log)
                f.write(log + '\n \n \n')

                if args.ft2d.p2:
                    print('step %d Protocol #2   (MPJPE) action-wise average J_Best: %f mm' % (
                    ii, errors_p2_actionwise[ii].item()))
                    f.write('step %d Protocol #2   (MPJPE) action-wise average J_Best: %f mm\n' % (
                    ii, errors_p2_actionwise[ii].item()))
                    print('step %d Protocol #2   (MPJPE) action-wise average P_Best: %f mm' % (
                        ii, errors_p2_actionwise_h[ii].item()))
                    f.write('step %d Protocol #2   (MPJPE) action-wise average P_Best: %f mm\n' % (
                        ii, errors_p2_actionwise_h[ii].item()))
                    print('step %d Protocol #2   (MPJPE) action-wise average P_Agg: %f mm' % (
                        ii, errors_p2_actionwise_mean[ii].item()))
                    f.write('step %d Protocol #2   (MPJPE) action-wise average P_Agg: %f mm\n' % (
                        ii, errors_p2_actionwise_mean[ii].item()))
                    print('step %d Protocol #2   (MPJPE) action-wise average J_Agg: %f mm' % (
                        ii, errors_p2_actionwise_select[ii].item()))
                    f.write('step %d Protocol #2   (MPJPE) action-wise average J_Agg: %f mm\n' % (
                        ii, errors_p2_actionwise_select[ii].item()))
            f.close()

        if not args.general.by_subject:
            run_evaluation(all_actions, action_filter)
        else:
            for subject in all_actions_by_subject.keys():
                print('Evaluating on subject', subject)
                run_evaluation(all_actions_by_subject[subject], action_filter)
                print('')
    if not args.general.nolog:
        writer.close()


if __name__ == "__main__":
    main()
