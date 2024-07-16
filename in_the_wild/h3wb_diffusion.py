import os
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import json

import torch.nn as nn

from common.camera import *
from common.generators import *
from common.loss import *
from in_the_wild.utils import Timer, evaluate_diffusion, add_path
from common.diffusionpose import D3DP
from common.h3wb_dataset import Human3WBDataset

# from joints_detectors.openpose.main import generate_kpts as open_pose

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# record time
def ckpt_time(ckpt=None):
    if not ckpt:
        return time.time()
    else:
        return time.time() - float(ckpt), time.time()


time0 = ckpt_time()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(args: DictConfig):

    dir_name = os.path.dirname(args.in_the_wild.video_path)
    basename = os.path.basename(args.in_the_wild.video_path)
    video_name = basename[:basename.rfind('.')]
    viz_output = f'{dir_name}/openpifpaf_{video_name}.mp4'

    save_path_dir = f'outputs/{video_name}'
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)

    print('Loading dataset...')
    dataset_path = 'data/train_' + args.data.dataset + '.npz'
    dataset = Human3WBDataset(dataset_path)

    keypoints_symmetry = dataset.keypoints_metadata['keypoints_symmetry']  # metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = kps_left, kps_right  # list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

    model_pos = D3DP(args, joints_left, joints_right, dataset=dataset, is_train=False,
         num_proposals=args.ft2d.num_proposals, sampling_timesteps=args.ft2d.sampling_timesteps)
    model_pos = nn.DataParallel(model_pos)
    model_pos = model_pos.cuda()

    # load kps
    kps = []
    with open(f'{dir_name}/{basename}.openpifpaf.json', 'r') as file:
        for line in file:
            kps.append(json.loads(line))

    keypoints = np.zeros((len(kps), args.data.num_kps, 2), dtype=np.float32)
    for ind, kp in enumerate(kps):
        pred = kp['predictions'][0]['keypoints']
        keypoints[ind, 1:, 0] = pred[::3]
        keypoints[ind, 1:, 1] = pred[1::3]
        static_joint = (keypoints[ind, 12:13, :] + keypoints[ind, 13:14, :]) / 2.
        keypoints[ind, :1, :] = static_joint

    # normlization keypoints  Suppose using the camera parameter
    import cv2
    cap = cv2.VideoCapture(args.in_the_wild.video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #keypoints = normalize_screen_coordinates(keypoints[..., :2], w=1000, h=1002)
    keypoints = normalize_screen_coordinates(keypoints[..., :2], w=frame_width, h=frame_height)

    ckpt, time1 = ckpt_time(time0)
    print('-------------- load data spends {:.2f} seconds'.format(ckpt))

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # cv2.imshow('window-name', frame)
        cv2.imwrite(f'outputs/{video_name}/frame_{count}.jpg', frame)
        count = count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()  # destroy all opened windows

    chk_filename = os.path.join(args.general.checkpoint, args.general.resume if args.general.resume else args.general.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos.load_state_dict(checkpoint['model_pos'], strict=False)

    ckpt, time2 = ckpt_time(time1)
    print('-------------- load 3D model spends {:.2f} seconds'.format(ckpt))

    #  Receptive field: 243 frames for args.arc [3, 3, 3, 3, 3]
    receptive_field = args.model.number_of_frames
    pad = (receptive_field - 1) // 2  # Padding on each side
    causal_shift = 0

    print('Rendering...')
    input_keypoints = keypoints.copy()
    print(input_keypoints.shape)
    gen = UnchunkedGenerator_Seq(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.model.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

    prediction = evaluate_diffusion(args, dataset, gen, model_pos, return_predictions=True, receptive_field=receptive_field,
                                    bs=args.model.batch_size) # b, t, h, 243, j, c
    b_sz, t_sz, h_sz, f_sz, j_sz, c_sz = prediction.shape
    total_frame = input_keypoints.shape[0]
    prediction2 = np.empty((t_sz, h_sz, total_frame, 134, 3)).astype(np.float32)
    ### reshape prediction as ground truth
    if total_frame / receptive_field > total_frame // receptive_field:
        batch_num = (total_frame // receptive_field) + 1
        for i in range(batch_num - 1):
            prediction2[:, :, i * receptive_field:(i + 1) * receptive_field, :, :] = prediction[i, :, :, :, :, :]
        left_frames = total_frame - (batch_num - 1) * receptive_field
        prediction2[:, :, -left_frames:, :, :] = prediction[-1, :, :, -left_frames:, :, :]
        # prediction = prediction2
    elif total_frame / receptive_field == total_frame // receptive_field:
        batch_num = (total_frame // receptive_field)
        for i in range(batch_num):
            prediction2[:, :, i * receptive_field:(i + 1) * receptive_field, :, :] = prediction[i, :, :, :, :, :]

    # save 3D joint points
    np.save(f'outputs/{video_name}/test_3d_{video_name}_output.npy', prediction2, allow_pickle=True)

    prediction2_c = prediction2.copy()
    rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
    prediction2 = camera_to_world(prediction2, R=rot, t=0)

    # We don't have the trajectory, but at least we can rebase the height
    prediction2[:, :, :, :, 2] -= np.min(prediction2[:, :, :, :, 2])

    np.save(f'outputs/{video_name}/test_3d_output_{video_name}_postprocess.npy', prediction2, allow_pickle=True)

    anim_output = {'Ours': prediction}
    input_keypoints = image_coordinates(input_keypoints[..., :2], w=1000, h=1002)

    ckpt, time3 = ckpt_time(time2)
    print('-------------- generate reconstruction 3D data spends {:.2f} seconds'.format(ckpt))

    if not viz_output:
        viz_output = 'outputs/alpha_result.mp4'

    from in_the_wild.visualization import draw_3d_image
    # render_animation(input_keypoints, anim_output,
    #                  Skeleton(), 25, args.viz_bitrate, np.array(70., dtype=np.float32), viz_output,
    #                  limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
    #                  input_video_path=args.viz_video, viewport=(1000, 1002),
    #                  input_video_skip=args.viz_skip)

    draw_3d_image(prediction2, dataset.skeleton(), np.array(70., dtype=np.float32), video_name)

    ckpt, time4 = ckpt_time(time3)
    print('total spend {:2f} second'.format(ckpt))


if __name__ == '__main__':
    main()
