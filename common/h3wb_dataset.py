# @author Nermin Samet


import os
import numpy as np
import copy

from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates
from common.h36m_dataset import h36m_cameras_intrinsic_params, h36m_cameras_extrinsic_params


class Human3WBDataset(MocapDataset):
    def __init__(self, path, add_root=True):

        # Load serialized dataset
        check_data = np.load(path, allow_pickle=True)
        self.metadata = check_data['metadata'].item()
        train_data = check_data['train_data'].item()

        # load private test set - S8
        test_data_3d = np.load(os.path.join(os.path.dirname(path), 'task1_test_3d.npz'), allow_pickle=True)['data'].item()
        train_data.update(test_data_3d)

        # prepare skeleton
        joints_left = self.metadata['left_side']
        joints_right = self.metadata['right_side']
        dups = []
        # filter duplicate kps in both sides
        for kp in joints_left:
            if kp in joints_right:
                dups.append(kp)
        offset = 0
        if add_root:
            offset = 1
        joints_left = [ele + offset for ele in joints_left if ele not in dups]
        joints_right = [ele + offset for ele in joints_right if ele not in dups]

        all_kps = []
        self.kps_order = ['body', 'left_foot', 'right_foot', 'face', 'left_hand', 'right_hand']
        for ind, kp in enumerate(self.kps_order):
            all_kps.extend(self.metadata[kp])
        all_kps = [x - 1 for x in all_kps]

        self.prepare_skeleton(root_added=add_root)

        # indices of roots for each body part
        self.root_indices = {
            'body': 0,
            'face': 54,
            'left_hand': 92,
            'right_hand': 113,
        }

        # indices of body joints that must match parts roots
        self.parts_connection_indices = {
            'face': 1,
            'left_hand': 10,
            'right_hand': 11,
        }

        self.num_kps = len(self.parents)  # if we add root num kps becomes 134 else it is 133
        self.keypoints_metadata = {'layout_name': 'h3wb',
                                   'num_joints': self.num_kps,
                                   'keypoints_symmetry': [joints_left, joints_right]
                                   }

        h36m_skeleton = Skeleton(parents=self.parents,
                                 joints_left=joints_left,
                                 joints_right=joints_right)

        self.predefined_one_hot_vec = np.zeros((len(self.kps_order), self.num_kps, 1), dtype=np.float32)
        for ind, kp in enumerate(self.kps_order):
            one_hot = np.zeros((self.keypoints_metadata['num_joints'], 1), dtype=np.float32)
            one_hot[self.metadata[kp]] = 1.
            self.predefined_one_hot_vec[ind, :] = one_hot

        # Deformable grouping
        # self.deformable_kps_order = ['body', 'legs_feets', 'arms', 'hands', 'face']
        # self.predefined_one_hot_vec_deformable = np.zeros((len(self.deformable_kps_order), self.num_kps, 1), dtype=np.float32)
        # h1 = [5, 6, 11, 12]  # body
        # h2 = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # foot
        # h3 = [7, 8, 9, 10]  # arms
        # h4 = [0, 1, 2, 3, 4] + self.metadata['face']  # face
        # h5 = self.metadata['left_hand'] + self.metadata['right_hand']  # hands
        #
        # for ind, kp in enumerate([h1, h2, h3, h4, h5]):
        #     one_hot = np.zeros((self.keypoints_metadata['num_joints'], 1), dtype=np.float32)
        #     one_hot[kp] = 1.
        #     self.predefined_one_hot_vec_deformable[ind, :] = one_hot

        super().__init__(fps=50, skeleton=h36m_skeleton)
        # we can use the same camera parameters since the intrinsics are the same!!
        self._cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
        for cameras in self._cameras.values():
            for i, cam in enumerate(cameras):
                cam.update(h36m_cameras_intrinsic_params[i])
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')

                # Normalize camera frame
                cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype(
                    'float32')
                cam['focal_length'] = cam['focal_length'] / cam['res_w'] * 2
                if 'translation' in cam:
                    cam['translation'] = cam['translation'] / 1000  # mm to meters

                # Add intrinsic parameters vector
                cam['intrinsic'] = np.concatenate((cam['focal_length'],
                                                   cam['center'],
                                                   cam['radial_distortion'],
                                                   cam['tangential_distortion']))

        self.camera_order_id = ['54138969', '55011271', '58860488', '60457274']
        self._data = {}
        self._cameras_full_data = {}

        if add_root:
            train_data = self.add_root_joint(train_data)

        for subject, actions in train_data.items():
            self._data[subject] = {}
            self._cameras_full_data[subject] = [self.metadata[subject][cam] for cam in self.camera_order_id]
            for action_name, act_data in actions.items():
                self._data[subject][action_name] = {
                    'positions': act_data['global_3d'].squeeze(),  # global coord
                    'cameras': [self.metadata[subject][cam] for cam in self.camera_order_id],
                    'positions_3d': [act_data[cam]['camera_3d'].squeeze() for cam in self.camera_order_id],
                    'pose_2d': [act_data[cam]['pose_2d'].squeeze() for cam in self.camera_order_id]
                }

        self.compute_part_joint_indices()

        print('Dataset preparation is done!')

    def prepare_skeleton(self, root_added=False):
        left_foot_parents = [15, 15, 15]
        right_foot_parents = [16, 16, 16]
        left_hand_parents = [9, 91, 92, 93, 94, 91, 96, 97, 98, 91, 100, 101, 102, 91, 104, 105, 106, 91, 108, 109, 110]
        right_hand_parents = [10, 112, 113, 114, 115, 112, 117, 118, 119, 112, 121, 122, 123, 112, 125, 126, 127, 112,
                              129, 130, 131]

        if root_added:
            # body_parents = [-1, 0, 1, 1, 1, 1, 0, 0, 6, 7, 8, 9, 0, 0, 12, 13, 14, 15]  # correct parents
            body_parents = [-1, -1, -1, -1, -1, -1, 0, 0, 6, 7, 8, 9, 0, 0, 12, 13, 14, 15]  # for face we can put just dots no need to draw lines!
            # body_parents = [-1, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 6, 7, 13, 14, 15, 15]
            # face_parents = [1] * len(self.metadata["face"])  # nose, for drawing lines
            face_parents = [-1] * len(self.metadata["face"])  # no root to draw points

            # shift all by 1
            left_foot_parents = [ele + 1 for ele in left_foot_parents]
            right_foot_parents = [ele + 1 for ele in right_foot_parents]
            left_hand_parents = [ele + 1 for ele in left_hand_parents]
            right_hand_parents = [ele + 1 for ele in right_hand_parents]
        else:
            body_parents = [-1, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]
            face_parents = [0] * len(self.metadata["face"])  # nose

        self.parents = body_parents + left_foot_parents + right_foot_parents + face_parents + left_hand_parents + right_hand_parents

    @staticmethod
    def add_root_joint(train_data, root_joint_inds=[11, 12]):
        # lets add a static joint to become root of all body joints between 11 and 12 joints (left and right hips)
        for subject, actions in train_data.items():
            for kk, vv in actions.items():
                for kkk, vvv in vv.items():
                    if kkk == 'frame_id':
                        continue
                    elif kkk == 'global_3d':
                        global_3d = vvv
                        d_shape = global_3d.shape
                        new_global_3d = np.zeros((d_shape[0], d_shape[1]+1, d_shape[2]))
                        new_global_3d[:, 1:, :] = global_3d
                        static_joint = (global_3d[:, root_joint_inds[0]:root_joint_inds[1], :] +
                                        global_3d[:, root_joint_inds[1]:root_joint_inds[1] + 1, :]) / 2.
                        new_global_3d[:, :1, :] = static_joint
                        train_data[subject][kk][kkk] = new_global_3d
                    else:
                        for cam_data_k, cam_data in vvv.items():
                            if cam_data_k == 'sample_id':
                                continue
                            global_3d = cam_data
                            d_shape = global_3d.shape
                            new_global_3d = np.zeros((d_shape[0], d_shape[1] + 1, d_shape[2]))
                            new_global_3d[:, 1:, :] = global_3d
                            static_joint = (global_3d[:, root_joint_inds[0]:root_joint_inds[1], :] +
                                            global_3d[:, root_joint_inds[1]:root_joint_inds[1] + 1, :]) / 2.
                            new_global_3d[:, :1, :] = static_joint
                            train_data[subject][kk][kkk][cam_data_k] = new_global_3d

        return train_data

    def supports_semi_supervised(self):
        return True

    def compute_part_joint_indices(self):
        parts = [
            "body", "face", "left_hand", "right_hand", "left_foot", "right_foot"
        ]
        self.parts_joint_indices = {
            part: [ele + 1 for ele in self.metadata[part]]
            for part in parts
        }
        self.parts_joint_indices["body"] = (
            [0] +
            self.parts_joint_indices["body"] +
            self.parts_joint_indices["left_foot"] +
            self.parts_joint_indices["right_foot"]
        )
        del self.parts_joint_indices["left_foot"]
        del self.parts_joint_indices["right_foot"]

