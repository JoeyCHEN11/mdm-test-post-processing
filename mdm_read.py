import numpy as np
import torch
import pytorch3d.transforms

def load_mdm_to_smplx_params(npy_path: str, device: str = 'cpu') -> dict:
    """
    读取 MDM 生成的 .npy 文件，并将其转换为 SMPL-X 模型的标准输入参数。
    
    Args:
        npy_path (str): .npy 文件路径
        device (str): 目标设备 ('cpu' or 'cuda')
    
    Returns:
        dict: 包含 SMPL-X 参数的字典，所有值均为 Tensor，形状为 (Frames, ...):
            - global_orient: (Frames, 1, 3) 
            - body_pose:     (Frames, 21, 3)
            - transl:        (Frames, 3)
            - betas:         (1, 10)  # 默认为全0
    """
    
    # 1. 加载 numpy 数据
    # fix for numpy 2.0 comptibility inside legacy pickle files just in case
    if not hasattr(np, 'infty'):
        np.infty = np.inf
        
    print(f"[Data]正在加载动作文件: {npy_path}")
    data = np.load(npy_path, allow_pickle=True).item()
    
    # 2. 提取并标准化维度 -> (Frames, Joints, 6)
    raw_thetas = data['thetas'] 
    
    # 处理 Batch 和 Channel 维度，目标: [Frames, 24, 6]
    if len(raw_thetas.shape) == 4: # (Batch, Joints, 6, Frames)
        # 取 Batch 0
        thetas_tensor = torch.from_numpy(raw_thetas[0]).permute(2, 0, 1).float()
    elif len(raw_thetas.shape) == 3: # (Joints, 6, Frames)
        thetas_tensor = torch.from_numpy(raw_thetas).permute(2, 0, 1).float()
    else:
        raise ValueError(f"未知的 thetas 形状: {raw_thetas.shape}")

    # 截取前 24 个关节 (SMPL 标准)
    if thetas_tensor.shape[1] > 24:
        thetas_tensor = thetas_tensor[:, :24, :]
        
    # 处理位移: (3, Frames) -> (Frames, 3)
    transl_tensor = torch.from_numpy(data['root_translation']).permute(1, 0).float()
    
    # 3. 转换旋转格式: 6D -> Matrix -> Axis-Angle
    # 优化任务通常基于 Axis-Angle 或 Matrix，这里转为 Axis-Angle 以适配 SMPL-X forward
    with torch.no_grad():
        rot_mats = pytorch3d.transforms.rotation_6d_to_matrix(thetas_tensor)
        rot_axis_angle = pytorch3d.transforms.matrix_to_axis_angle(rot_mats) # Shape: [Frames, 24, 3]

    num_frames = rot_axis_angle.shape[0]

    # 4. 拆分参数以适配 SMPL-X Forward 签名
    # SMPL-X 定义:
    # 0号关节: root (global_orient)
    # 1-21号关节: body (body_pose)
    # 22-23号关节: smpl hands (通常在 smplx 中被忽略或作为 hand_pose 的一部分，这里为了保持身体姿态，我们取前22个)
    
    global_orient = rot_axis_angle[:, 0:1, :] # [Frames, 1, 3]
    body_pose = rot_axis_angle[:, 1:22, :]    # [Frames, 21, 3]

    # 5. 转移到设备
    device_obj = torch.device(device)
    output_params = {
        'global_orient': global_orient.to(device_obj),
        'body_pose': body_pose.to(device_obj),
        'transl': transl_tensor.to(device_obj),
        'betas': torch.zeros((1, 10), device=device_obj).float() # 默认体型
    }

    print(f"[Data] 数据加载完成。帧数: {num_frames}, 设备: {device}")
    return output_params