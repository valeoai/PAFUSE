import torch
import numpy as np
import hashlib


def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result


def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value


def load_pretrained_weights(model, checkpoint):
    """Load pretrianed weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - weight_path (str): path to pretrained weights
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    # model.state_dict(model_dict).requires_grad = False
    return model


def center_pose_at_root(pose_3d, root_idx=0, joint_dim=-2, revert=False):
    """ Generic function translating arbitrary 3D poses so that the root joint
    is located at (0,0,0)
    """
    assert joint_dim < len(pose_3d.shape)
    assert root_idx < pose_3d.shape[joint_dim]

    # offset = root positions
    offset = pose_3d.select(dim=joint_dim, index=root_idx).unsqueeze(joint_dim)

    if revert:
        offset *= -1

    return pose_3d - offset


def center_pose_parts(pose_3d, dataset):
    """ Generic function translating each part (body, face, left/right hands)
    of an arbitrary 3D pose so that their corresponding root joints are
    located at (0,0,0)
    """
    parts_joint_indices = dataset.parts_joint_indices
    parts_root_indices = dataset.root_indices
    pose_3d_ret = torch.zeros_like(pose_3d)

    for part, joint_indices in parts_joint_indices.items():
        pose_3d_ret[..., joint_indices, :] = center_pose_at_root(
            pose_3d,
            root_idx=parts_root_indices[part],
            joint_dim=-2,  # Fixed here, as it seem to always be the case
        )[..., joint_indices, :]
    return pose_3d_ret


def wb_pose_from_parts(part_based_pose, dataset):
    parts_joint_indices = dataset.parts_joint_indices
    root_connection_indices = dataset.parts_connection_indices
    root_connection_indices.update({'body': 0})
    part_based_pose_ret = torch.zeros_like(part_based_pose)
    for part, joint_indices in parts_joint_indices.items():
        if part in root_connection_indices:
            part_based_pose_ret[..., joint_indices, :] = center_pose_at_root(
                part_based_pose,
                root_idx=root_connection_indices[part],
                joint_dim=-2,  # Fixed here, as it seem to always be the case
                revert=True,
            )[..., joint_indices, :]
    return part_based_pose_ret


def test_funcs(dataset):
    """
    Verifying partition and wholebody construction funcs on sample data.
    """
    input3d_test = torch.ones((1, 1, 134, 3), dtype=torch.float32)

    input3d_test[:, :, 1, :] = 2.
    input3d_test[:, :, 10, :] = 5.
    input3d_test[:, :, 11, :] = 13.
    input3d_testc = input3d_test.clone()
    # gt part based split
    gt_parted_input = input3d_test.clone()
    gt_parted_input[:, :, dataset.parts_joint_indices['body'], :] = 0.
    gt_parted_input[:, :, 1, :] = 1.
    gt_parted_input[:, :, 10, :] = 4.
    gt_parted_input[:, :, 11, :] = 12.
    gt_parted_input[:, :, dataset.parts_joint_indices['face'], :] = -1.
    gt_parted_input[:, :, dataset.parts_joint_indices['left_hand'], :] = -4.
    gt_parted_input[:, :, dataset.parts_joint_indices['right_hand'], :] = -12.

    # first test centering pose parts.
    input3d_test = center_pose_parts(input3d_test, dataset)
    assert torch.sum(input3d_test - gt_parted_input) == 0

    # then test retrieving the parted data to trajectory
    input3d_test = wb_pose_from_parts(input3d_test, dataset)
    # calculate trajectory on input, and compare with retrieved wb pose
    input3d_testc = center_pose_at_root(input3d_testc)
    assert torch.sum(input3d_test - input3d_testc) == 0
