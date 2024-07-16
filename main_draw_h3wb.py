# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import torch.nn as nn
import os
import sys
import errno

from common.camera import *
from common.diffusionpose import *

from common.loss import *
from common.generators import UnchunkedGenerator_Seq
from common.utils import *
from common.logging import Logger
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

from common.h3wb_dataset import Human3WBDataset
from common.utils import center_pose_at_root, center_pose_parts, wb_pose_from_parts

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig):
    # cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.experiment.gpu

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
    print('python ' + ' '.join(sys.argv))
    print("CUDA Device Count: ", torch.cuda.device_count())
    print(args)

    manualSeed = 1
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    # if not assign checkpoint path, Save checkpoint file into log folder
    if args.general.checkpoint == '':
        args.general.checkpoint = args.general.log + '_' + TIMESTAMP
    try:
        # Create checkpoint directory if it does not exist
        os.makedirs(args.general.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', args.general.checkpoint)

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

    def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
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

        stride = args.experiment.downsample
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


    action_filter = None if args.data.actions == '*' else args.data.actions.split(',')
    if action_filter is not None:
        print('Selected actions:', action_filter)

    cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

    # set receptive_field as number assigned
    receptive_field = args.model.number_of_frames
    print('INFO: Receptive field: {} frames'.format(receptive_field))
    if not args.general.nolog:
        writer.add_text(args.general.log + '_' + TIMESTAMP + '/Receptive field', str(receptive_field))
    pad = (receptive_field - 1) // 2  # Padding on each side
    min_loss = args.model.min_loss
    width = cam['res_w']
    height = cam['res_h']
    num_joints = keypoints_metadata['num_joints']

    print('INFO: Creating the models')
    model_pos_train = D3DP(args, joints_left, joints_right, dataset=dataset, is_train=True)
    model_pos_test_temp = D3DP(args, joints_left, joints_right, dataset=dataset, is_train=False)
    model_pos = D3DP(args, joints_left, joints_right, dataset=dataset, is_train=False,
                     num_proposals=args.ft2d.num_proposals, sampling_timesteps=args.ft2d.sampling_timesteps)

    #################
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


    # Evaluate
    def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False):
        epoch_loss_3d_pos = torch.zeros(args.ft2d.sampling_timesteps).cuda()
        epoch_loss_3d_pos_mean = torch.zeros(args.ft2d.sampling_timesteps).cuda()
        epoch_loss_3d_pos_procrustes = 0
        epoch_loss_3d_pos_scale = 0
        epoch_loss_3d_vel = 0
        with torch.no_grad():
            model_eval = model_pos
            if not use_trajectory_model:
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
            for _, batch, batch_2d in test_generator.next_epoch():
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                inputs_3d = torch.from_numpy(batch.astype('float32'))

                inputs_2d_flip = inputs_2d.clone()
                inputs_2d_flip[:, :, :, 0] *= -1
                inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

                ##### convert size
                inputs_3d_p = inputs_3d
                inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d_p)
                inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d_p)

                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                    inputs_2d_flip = inputs_2d_flip.cuda()
                    inputs_3d = inputs_3d.cuda()

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
                    else:
                        inputs_2d_single = inputs_2d[batch_cnt * bs:(batch_cnt + 1) * bs]
                        inputs_2d_flip_single = inputs_2d_flip[batch_cnt * bs:(batch_cnt + 1) * bs]
                        inputs_3d_single = inputs_3d[batch_cnt * bs:(batch_cnt + 1) * bs]

                    predicted_3d_pos_single = model_eval(inputs_2d_single, inputs_3d_single,
                                                         input_2d_flip=inputs_2d_flip_single)  # b, t, h, f, j, c

                    if args.general.part_based_model:
                        # NOTE: moved below lines from diff model to here since they are post-processing steps.
                        predicted_3d_pos_single = wb_pose_from_parts(predicted_3d_pos_single, dataset=dataset)
                        inputs_3d_single = wb_pose_from_parts(inputs_3d_single, dataset=dataset)

                    if return_predictions:
                        if batch_cnt == 0:
                            out_all = predicted_3d_pos_single.cpu().numpy()
                        else:
                            out_all = np.concatenate((out_all, predicted_3d_pos_single.cpu().numpy()), axis=0)

                    if quickdebug:
                        if N == inputs_3d.shape[0] * inputs_3d.shape[1]:
                            break

                return out_all

        log_path = os.path.join(args.checkpoint, 'test_log_h%d_t%d.txt' % (args.num_proposals, args.sampling_timesteps))
        f = open(log_path, mode='a')
        if action is None:
            print('----------')
        else:
            print('----' + action + '----')
            f.write('----' + action + '----\n')

        e1 = (epoch_loss_3d_pos / N) * 1000
        e1_mean = (epoch_loss_3d_pos_mean / N) * 1000

        print('Test time augmentation:', test_generator.augment_enabled())
        for ii in range(e1.shape[0]):
            print('step %d : Protocol #1 Error (MPJPE):' % ii, e1[ii].item(), 'mm')
            f.write('step %d : Protocol #1 Error (MPJPE): %f mm\n' % (ii, e1[ii].item()))
            print('step %d : Protocol #1 Error (MPJPE) mean pose:' % ii, e1_mean[ii].item(), 'mm')
            f.write('step %d : Protocol #1 Error (MPJPE) mean pose: %f mm\n' % (ii, e1_mean[ii].item()))

        print('----------')
        f.write('----------\n')

        f.close()

        return e1, e1_mean

    if args.general.render:
        print('Rendering...')

        input_keypoints = keypoints[args.viz.viz_subject][args.viz.viz_action][args.viz.viz_camera].copy()
        ground_truth = None
        if args.viz.viz_subject in dataset.subjects() and args.viz.viz_action in dataset[args.viz.viz_subject]:
            if 'positions_3d' in dataset[args.viz.viz_subject][args.viz.viz_action]:
                ground_truth = dataset[args.viz.viz_subject][args.viz.viz_action]['positions_3d'][args.viz.viz_camera].copy().astype('float32')
        if ground_truth is None:
            print('INFO: this action is unlabeled. Ground truth will not be rendered.')

        gen = UnchunkedGenerator_Seq(None, [ground_truth], [input_keypoints],
                                     pad=pad, causal_shift=causal_shift, augment=args.model.test_time_augmentation,
                                     kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                     joints_right=joints_right)
        prediction = evaluate(gen, return_predictions=True)
        b_sz, t_sz, h_sz, f_sz, j_sz, c_sz = prediction.shape
        if args.viz.compare:
            from common.model_poseformer import PoseTransformer

            model_pf = PoseTransformer(num_frame=81, num_joints=17, in_chans=2, num_heads=8, mlp_ratio=2., qkv_bias=False,
                                       qk_scale=None, drop_path_rate=0.1)
            if torch.cuda.is_available():
                model_pf = nn.DataParallel(model_pf)
                model_pf = model_pf.cuda()
            prediction_pf = evaluate(gen, newmodel=model_pf, return_predictions=True)

        # if model_traj is not None and ground_truth is None:
        #     prediction_traj = evaluate(gen, return_predictions=True, use_trajectory_model=True)
        #     prediction += prediction_traj
        prediction2 = np.empty((t_sz, h_sz, ground_truth.shape[0], args.data.num_kps, 3)).astype(np.float32)
        ### reshape prediction as ground truth
        if ground_truth.shape[0] / receptive_field > ground_truth.shape[0] // receptive_field:
            batch_num = (ground_truth.shape[0] // receptive_field) + 1
            for i in range(batch_num - 1):
                prediction2[:, :, i * receptive_field:(i + 1) * receptive_field, :, :] = prediction[i, :, :, :, :, :]
            left_frames = ground_truth.shape[0] - (batch_num - 1) * receptive_field
            prediction2[:, :, -left_frames:, :, :] = prediction[-1, :, :, -left_frames:, :, :]
            # prediction = prediction2
        elif ground_truth.shape[0] / receptive_field == ground_truth.shape[0] // receptive_field:
            batch_num = (ground_truth.shape[0] // receptive_field)
            for i in range(batch_num):
                prediction2[:, :, i * receptive_field:(i + 1) * receptive_field, :, :] = prediction[i, :, :, :, :, :]

        if args.viz.viz_export is not None:
            print('Exporting joint positions to', args.viz.viz_export)
            # Predictions are in camera space
            np.save(args.viz.viz_export, prediction)

        f_all_sz = ground_truth.shape[0]
        if ground_truth is not None:
            # Reapply trajectory
            trajectory = ground_truth[:, :1]
            # ground_truth[:, 1:] += trajectory
            trajectory = trajectory.reshape(1, 1, f_all_sz, 1, 3)
            prediction2 += trajectory
            if args.viz.compare:
                prediction_pf += trajectory

        # Invert camera transformation
        cam = dataset.cameras()[args.viz.viz_subject][args.viz.viz_camera]
        if ground_truth is not None:
            if args.viz.compare:
                prediction_pf = camera_to_world(prediction_pf, R=cam['orientation'], t=cam['translation'])
            aa = prediction2[0, 0]
            bb = camera_to_world(aa, R=cam['orientation'], t=cam['translation'])
            prediction2_world = camera_to_world(prediction2, R=cam['orientation'], t=cam['translation'])
            ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
        else:
            # If the ground truth is not available, take the camera extrinsic params from a random subject.
            # They are almost the same, and anyway, we only need this for visualization purposes.
            for subject in dataset.cameras():
                if 'orientation' in dataset.cameras()[subject][args.viz.viz_camera]:
                    rot = dataset.cameras()[subject][args.viz.viz_camera]['orientation']
                    break
            if args.viz.compare:
                prediction_pf = camera_to_world(prediction_pf, R=rot, t=0)
                prediction_pf[:, :, 2] -= np.min(prediction_pf[:, :, 2])
            prediction = camera_to_world(prediction, R=rot, t=0)
            # We don't have the trajectory, but at least we can rebase the height
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])

        prediction2_reshape = prediction2.reshape(t_sz * h_sz * prediction2.shape[2], j_sz, 3)
        # cam_intri = cam['intrinsic'][None].repeat(prediction2_reshape.shape[0],1)
        prediction2_reshape = torch.from_numpy(prediction2_reshape)
        cam_intri = torch.from_numpy(np.repeat(cam['intrinsic'][None], prediction2_reshape.shape[0], axis=0))
        poses_2d_reproj = project_to_2d(prediction2_reshape, cam_intri)
        poses_2d_reproj = poses_2d_reproj.reshape(t_sz, h_sz, prediction2.shape[2], j_sz, 2)

        input_keypoints_pix = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])
        poses_2d_reproj_pix = image_coordinates(poses_2d_reproj[..., :2].numpy(), w=cam['res_w'], h=cam['res_h'])

        parents = dataset.skeleton().parents()
        # data_x = data_all[:,2,0]
        # data_y = data_all[:,2,1]
        # data_x_re = poses_2d_reproj[:,2,0]
        # data_y_re = poses_2d_reproj[:,2,1]
        plt_path = f'./plot/{args.data.dataset}_{args.viz.viz_subject}_{args.viz.viz_action}_{args.viz.viz_camera}/'
        if not os.path.isdir(plt_path):
            os.makedirs(plt_path)

        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        # draw 2d reprojection in a square image

        # for f in range(poses_2d_reproj.shape[2]):
        #     if f % 5 != 0:
        #         continue
        #     frame = f
        #     data = input_keypoints[frame]
        #     data_re = poses_2d_reproj[:, :, frame]
        #
        #     for t in range(data_re.shape[0] - 1, data_re.shape[0]):
        #
        #         # ax = plt.gca()
        #         # ax.xaxis.set_ticks_position('top')
        #         # ax.invert_yaxis()
        #
        #         # fig, ax = plt.subplots()
        #         # fig.tight_layout()
        #         #
        #         # # Override default padding
        #         # fig.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])
        #
        #         #plt.axis("off")
        #         plt.xlim(0,cam['res_w'])
        #         plt.ylim(0,cam['res_h'])
        #         # plt.xlim(200,800)
        #         # plt.ylim(200,800)
        #         # plt.xlim(-1, 1)
        #         # plt.ylim(-1, 1)
        #         # plt.imshow(image)
        #
        #         #plt.scatter(data[:, 0], data[:, 1], s=1, c='blue')
        #         # for i in range(17):
        #         #     #plt.text(s=str(i), x=data[i,0], y=data[i,1]-10,color='black', fontsize='3')
        #         #     plt.text(s=str(i), x=data[i, 0], y=data[i, 1] - 0.02, color='blue', fontsize='3')
        #
        #         import matplotlib.colors as mcolors
        #         colors = list(mcolors.TABLEAU_COLORS.keys())
        #
        #         for j, j_parent in enumerate(parents):
        #             if j_parent == -1:
        #                 continue
        #
        #             plt.plot([data[j, 0], data[j_parent, 0]],
        #                      [data[j, 1], data[j_parent, 1]], color='blue', linewidth=0.2)
        #
        #             for h in range(data_re.shape[1]):
        #                 plt.plot([data_re[t, h, j, 0], data_re[t, h, j_parent, 0]],
        #                          [data_re[t, h, j, 1], data_re[t, h, j_parent, 1]], linestyle='--', linewidth=0.2, c=mcolors.TABLEAU_COLORS[colors[h]])
        #
        #
        #         #plt.scatter(data_re[:, 0], data_re[:, 1], s=1, c='red')
        #         # for i in range(17):
        #         #     #plt.text(s=str(i), x=data[i,0], y=data[i,1]-10,color='black', fontsize='3')
        #         #     plt.text(s=str(i), x=data_re[i, 0], y=data_re[i, 1] - 0.02, color='red', fontsize='3')
        #
        #         # for j, j_parent in enumerate(parents):
        #         #     if j_parent == -1:
        #         #         continue
        #         #
        #         #     plt.plot([data_re[j, 0], data_re[j_parent, 0]],
        #         #              [data_re[j, 1], data_re[j_parent, 1]], color='red', linewidth=0.2)
        #
        #         ax = plt.gca()
        #         ax.xaxis.set_ticks_position('top')
        #         ax.invert_yaxis()
        #         ax.set_aspect('equal')
        #
        #         import matplotlib.patches as patches
        #         rect = patches.Rectangle((0, 0), 1000, 1000, linewidth=1, edgecolor='black', facecolor='none')
        #
        #         # Add the patch to the Axes
        #         ax.add_patch(rect)
        #
        #         plt.axis('off')
        #         plt.xticks([])
        #         plt.yticks([])
        #
        #         # plt.margins(x=0)
        #         # plt.margins(y=0)
        #         #fig, ax = plt.subplots()
        #
        #
        #         # plt.show()
        #
        #         #plt.savefig(plt_path + "/%s_%s_%d_frame%d_t%d_2d_zoom.png" % (args.viz_subject,args.viz_action,args.viz_camera,frame, t), bbox_inches="tight", pad_inches=0.0, dpi=300)
        #         plt.savefig(plt_path + "/%s_%s_%d_frame%d_t%d_2d_square.png" % (
        #         args.viz_subject, args.viz_action, args.viz_camera, frame, t), bbox_inches="tight", pad_inches=0.1, dpi=300)
        #
        #         plt.close()
        #         #print("")

        # for f in range(poses_2d_reproj.shape[2]):
        #     if f % 5 != 0:
        #         continue
        #     frame = f
        #     data = input_keypoints[frame]
        #     data_re = poses_2d_reproj[:, :, frame]
        #
        #     for t in range(data_re.shape[0] - 1, data_re.shape[0]):
        #
        #         # ax = plt.gca()
        #         # ax.xaxis.set_ticks_position('top')
        #         # ax.invert_yaxis()
        #
        #         fig, ax = plt.subplots(1)
        #
        #         # Set whitespace to 0
        #         #fig.subplots_adjust(left=0, right=0, bottom=0, top=0)
        #         # fig, ax = plt.subplots()
        #         # fig.tight_layout()
        #         #
        #         # # Override default padding
        #         # fig.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])
        #
        #         #plt.axis("off")
        #         ax.xlim(0,cam['res_w'])
        #         ax.ylim(0,cam['res_h'])
        #         # plt.xlim(200,800)
        #         # plt.ylim(200,800)
        #         # plt.xlim(-1, 1)
        #         # plt.ylim(-1, 1)
        #         # plt.imshow(image)
        #
        #         #plt.scatter(data[:, 0], data[:, 1], s=1, c='blue')
        #         # for i in range(17):
        #         #     #plt.text(s=str(i), x=data[i,0], y=data[i,1]-10,color='black', fontsize='3')
        #         #     plt.text(s=str(i), x=data[i, 0], y=data[i, 1] - 0.02, color='blue', fontsize='3')
        #
        #         import matplotlib.colors as mcolors
        #         colors = list(mcolors.TABLEAU_COLORS.keys())
        #
        #         for j, j_parent in enumerate(parents):
        #             if j_parent == -1:
        #                 continue
        #
        #             ax.plot([data[j, 0], data[j_parent, 0]],
        #                      [data[j, 1], data[j_parent, 1]], color='blue', linewidth=0.2)
        #
        #             for h in range(data_re.shape[1]):
        #                 ax.plot([data_re[t, h, j, 0], data_re[t, h, j_parent, 0]],
        #                          [data_re[t, h, j, 1], data_re[t, h, j_parent, 1]], linestyle='--', linewidth=0.2, c=mcolors.TABLEAU_COLORS[colors[h]])
        #
        #
        #         #plt.scatter(data_re[:, 0], data_re[:, 1], s=1, c='red')
        #         # for i in range(17):
        #         #     #plt.text(s=str(i), x=data[i,0], y=data[i,1]-10,color='black', fontsize='3')
        #         #     plt.text(s=str(i), x=data_re[i, 0], y=data_re[i, 1] - 0.02, color='red', fontsize='3')
        #
        #         # for j, j_parent in enumerate(parents):
        #         #     if j_parent == -1:
        #         #         continue
        #         #
        #         #     plt.plot([data_re[j, 0], data_re[j_parent, 0]],
        #         #              [data_re[j, 1], data_re[j_parent, 1]], color='red', linewidth=0.2)
        #
        #         #ax = plt.gca()
        #         ax.xaxis.set_ticks_position('top')
        #         ax.invert_yaxis()
        #         ax.set_aspect('equal')
        #         ax.xticks([])
        #         ax.yticks([])
        #
        #         # ax.margins(x=0)
        #         # ax.margins(y=0)
        #         #fig, ax = plt.subplots()
        #
        #
        #         # plt.show()
        #
        #         #plt.savefig(plt_path + "/%s_%s_%d_frame%d_t%d_2d_zoom.png" % (args.viz_subject,args.viz_action,args.viz_camera,frame, t), bbox_inches="tight", pad_inches=0.0, dpi=300)
        #         plt.savefig(plt_path + "/%s_%s_%d_frame%d_t%d_2d_zoom.png" % (
        #         args.viz_subject, args.viz_action, args.viz_camera, frame, t), bbox_inches="tight", pad_inches=0.1, dpi=300)
        #
        #         plt.close()
        #         #print("")

        from common.visualization import draw_3d_image, draw_3d_image_select, draw_3d_res_h3wb

        # draw_3d_res_h3wb(prediction2_world, ground_truth, dataset.skeleton(), cam['azimuth'], args.viz.viz_subject,
        #                      args.viz.viz_action, args.viz.viz_camera, input_keypoints, poses_2d_reproj)
        # draw_3d_image(prediction2_world, ground_truth, dataset.skeleton(), cam['azimuth'], args.viz.viz_subject, args.viz.viz_action, args.viz.viz_camera)

        draw_3d_image_select(prediction2_world, ground_truth, dataset.skeleton(), cam['azimuth'], args.viz.viz_subject,
                             args.viz.viz_action, args.viz.viz_camera, input_keypoints, poses_2d_reproj)

    if not args.general.nolog:
        writer.close()


if __name__ == "__main__":
    main()
