import numpy as np
import torch
import smplx
import pyrender
import trimesh
import time
import os

if not hasattr(np, 'infty'):
    np.infty = np.inf
# --- 1. 数学工具: 6D -> Matrix -> Axis-Angle ---

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """ 6D连续旋转 -> 3x3旋转矩阵 """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def robust_matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """ 
    3x3旋转矩阵 -> 轴角向量 (Axis-Angle)
    这是为了兼容 SMPLX forward 函数最稳健的输入格式
    """
    # 提取四元数 (Quaternions) 作为中间表达更稳定
    # 但为了代码简短，这里使用 PyTorch 官方推荐的计算方式的简化版
    # 如果你有 pytorch3d，可以直接用 pytorch3d.transforms.matrix_to_axis_angle
    
    # 这里我们使用一个简单的近似转换，或者利用 trace 计算
    # cos(theta) = (trace(R) - 1) / 2
    
    batch_dims = matrix.shape[:-2]
    R = matrix.reshape(-1, 3, 3)
    
    ac = (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2
    ac = torch.clamp(ac, -1.0, 1.0)
    theta = torch.acos(ac).unsqueeze(1)
    
    # 避免除以0 (对于微小旋转)
    eps = 1e-6
    sin_theta = torch.sin(theta)
    
    # 正常情况 axis = [R21-R12, R02-R20, R10-R01] / 2sin(theta)
    axis = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1]
    ], dim=1)
    
    # 归一化并乘以 theta
    axis_angle = axis / (2 * sin_theta + eps) * theta
    
    # 对于极小角度 (theta -> 0)，上述公式不稳定，直接设为0
    # 这是一个简化处理，对于渲染已经足够
    mask = sin_theta.squeeze() < eps
    axis_angle[mask] = 0.0
    
    return axis_angle.reshape(*batch_dims, 3)

# --- 2. 核心播放器类 ---

class MotionPlayerSMPLX:
    def __init__(self, npy_path, model_folder):
        print(f"[Info] 加载动作数据: {npy_path}")
        data = np.load(npy_path, allow_pickle=True).item()
        
        # A. 数据清洗
        raw_thetas = data['thetas'] 
        # 统一转为 (Frames, 24, 6)
        if len(raw_thetas.shape) == 4:
            self.thetas = torch.from_numpy(raw_thetas[0]).permute(2, 0, 1).float()
        elif len(raw_thetas.shape) == 3:
            self.thetas = torch.from_numpy(raw_thetas).permute(2, 0, 1).float()
            
        if self.thetas.shape[1] > 24:
            self.thetas = self.thetas[:, :24, :]
            
        self.trans = torch.from_numpy(data['root_translation']).permute(1, 0).float()
        self.num_frames = self.thetas.shape[0]

        # B. 初始化模型
        print(f"[Info] 加载 SMPL-X 模型 (Path: {model_folder})...")
        self.model = smplx.create(
            model_path=model_folder, 
            model_type='smplx', 
            gender='neutral', 
            use_pca=False,
            batch_size=1
        )
        
        # C. 预计算 (Matrix -> Axis-Angle -> Mesh)
        print("[Info] 正在计算蒙皮 (转换 Matrix -> Axis-Angle)...")
        self.meshes = []
        
        with torch.no_grad():
            # 1. 先转 Matrix (为了准确性)
            rot_mats = rotation_6d_to_matrix(self.thetas) # [Frames, 24, 3, 3]
            
            # 2. 再转 Axis-Angle (为了喂给 SMPL-X)
            # 这一步是修复 "165 vs 55" 错误的关键
            rot_axis_angle = robust_matrix_to_axis_angle(rot_mats) # [Frames, 24, 3]
            
            for i in range(self.num_frames):
                # 拆分参数
                # rot_axis_angle[i:i+1] shape is [1, 24, 3]
                
                global_orient = rot_axis_angle[i:i+1, 0:1, :] # [1, 1, 3]
                body_pose = rot_axis_angle[i:i+1, 1:22, :]    # [1, 21, 3]
                
                # 喂给模型
                output = self.model(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    transl=self.trans[i:i+1],
                    betas=torch.zeros(1, 10),
                    # 显式补全其他关节 (使用 Axis-Angle 的 0 向量: [1, N, 3])
                    jaw_pose=torch.zeros(1, 1, 3),
                    leye_pose=torch.zeros(1, 1, 3),
                    reye_pose=torch.zeros(1, 1, 3),
                    left_hand_pose=torch.zeros(1, 15, 3),
                    right_hand_pose=torch.zeros(1, 15, 3)
                )
                
                verts = output.vertices[0].cpu().numpy()
                mesh = trimesh.Trimesh(vertices=verts, faces=self.model.faces)
                mesh.visual.vertex_colors = [180, 180, 200, 255]
                self.meshes.append(mesh)

    def run(self):
        # 1. 设置场景
        scene = pyrender.Scene()
        # 初始化节点
        self.mesh_node = pyrender.Mesh.from_trimesh(self.meshes[0])
        self.node_ref = scene.add(self.mesh_node)
        
        # 灯光
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=np.eye(4))
        # 补光 (防止背部太黑)
        scene.add(pyrender.DirectionalLight(color=[0.5, 0.5, 0.5], intensity=1.0), 
                  pose=np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]]))
        
        print("[Info] 启动播放器 (按 'h' 查看帮助)...")
        
        # 2. 启动 Viewer 并捕获实例 (关键修正!)
        # run_in_thread=True 会让 Viewer 在后台运行，返回 Viewer 实例给主线程
        self.viewer = pyrender.Viewer(scene, run_in_thread=True, use_raymond_lighting=True)
        
        # 3. 动画循环
        print("[Info] 开始循环播放...")
        while self.viewer.is_active: # 当窗口关闭时自动退出循环
            for i in range(self.num_frames):
                # 如果窗口关了，就停止
                if not self.viewer.is_active:
                    break

                # --- 关键修正: 使用 Viewer 的锁 ---
                self.viewer.render_lock.acquire() 
                try:
                    # 删除旧帧
                    scene.remove_node(self.node_ref)
                    # 添加新帧
                    self.node_ref = scene.add(pyrender.Mesh.from_trimesh(self.meshes[i]))
                finally:
                    self.viewer.render_lock.release()
                
                # 控制播放速度 (30 FPS)
                time.sleep(1/30.0)

# --- 运行配置 ---
model_folder = './body_models' 
npy_file = 'data/mdm_smpl_res_hug.npy'

if os.path.exists(npy_file):
    try:
        player = MotionPlayerSMPLX(npy_file, model_folder)
        player.run()
    except Exception as e:
        print(f"[Error] 运行出错: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"[Error] 找不到数据文件: {npy_file}")