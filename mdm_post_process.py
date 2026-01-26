import torch
import torch.nn.functional as F
import smplx
import time
from coap import attach_coap
import numpy as np
import pyrender
import trimesh
from tqdm import tqdm
from mdm_read import load_mdm_to_smplx_params

# Fix for NumPy 2.0 compatibility
if not hasattr(np, 'infty'):
    np.infty = np.inf

def optimize_single_frame_coap(
    model, 
    init_params, 
    max_iters=50, 
    lr=0.01, 
    selfpen_weight=1.0, 
    pose_prior_weight=10000.0,
    stop_threshold=0.1, 
    verbose=False
):
    """
    对单帧 SMPL 参数进行 COAP 优化以去除自穿模。
    
    Args:
        model: 已 attach_coap 的 SMPL-X 模型实例
        init_params (dict): 包含 'global_orient', 'body_pose', 'transl', 'betas' 的 Tensor 字典
                            形状预期为 (1, ...) 即 batch size 为 1
        max_iters (int): 最大优化步数
        lr (float): 学习率
        selfpen_weight (float): 穿模惩罚权重
        pose_prior_weight (float): 保持原始姿态的权重
        stop_threshold (float): 早停阈值，当 loss_sp 小于此值时停止
        verbose (bool): 是否打印优化过程
        
    Returns:
        dict:包含优化后参数的字典 (detached tensors)
    """
    
    # 1. 复制并设置需优化的参数
    # 我们通常只微调 body_pose 来解开穿模，保持 global_orient (朝向) 和 transl (位移) 不变
    # 除非是为了解决身体与环境的碰撞（目前只考虑自穿模）
    
    # Clone 确保不影响原始数据，并允许梯度追踪
    body_pose = init_params['body_pose'].clone().detach().requires_grad_(True)
    
    # 其他参数保持固定
    global_orient = init_params['global_orient'].clone().detach()
    transl = init_params['transl'].clone().detach()
    betas = init_params['betas'].clone().detach()
    
    # 原始姿态参考 (用于计算偏移惩罚)
    ref_body_pose = init_params['body_pose'].clone().detach()
    
    # 2. 配置优化器
    optimizer = torch.optim.Adam([body_pose], lr=lr)
    
    # 3. 优化循环
    for i in range(max_iters):
        optimizer.zero_grad()
        
        # SMPL Forward
        # 注意：COAP 的 self_collision_loss 内部可能需要 full_pose 或 vertices
        # 这里我们按标准流程调用
        output = model(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            betas=betas,
            return_verts=True,
            return_full_pose=True
        )
        
        # 计算 Loss
        # A. 穿模 Loss
        loss_sp = model.coap.self_collision_loss(output)
        
        # B. 如果已经没有穿模，提前结束
        if loss_sp.item() < stop_threshold: 
            if verbose: print(f"    [Frame Opt] Early stop at iter {i}, loss_sp: {loss_sp.item():.5f}")
            break
            
        # C. 正则项 Loss (防止动作变形过大)
        loss_prior = F.mse_loss(body_pose, ref_body_pose)
        
        # 总 Loss
        loss = (loss_sp * selfpen_weight) + (loss_prior * pose_prior_weight)
        
        # 反向传播更新
        loss.backward()
        optimizer.step()
        
        if verbose:
            print(f"    [Frame Opt] Iter {i}: Total={loss.item():.4f} (SP={loss_sp.item():.4f})")
            
    # 4. 封装返回结果
    optimized_params = {
        'global_orient': global_orient.detach(),
        'body_pose': body_pose.detach(),
        'transl': transl.detach(),
        'betas': betas.detach()
    }
    
    del optimizer
    # del loss
    del output
    
    return optimized_params

def run_optimization_pipeline(npy_path, model_path, device='cuda'):
    print(f"[Pipeline] 1. Loading Data from {npy_path}...")
    # 加载原始数据
    original_params = load_mdm_to_smplx_params(npy_path, device=device)
    num_frames = original_params['body_pose'].shape[0]
    
    print(f"[Pipeline] 2. Initializing SMPL-X with COAP...")
    
    # 准备容器存储优化后的每一帧
    optimized_results = {
        'global_orient': [],
        'body_pose': [],
        'transl': [],
        'betas': []
    }
    # 初始化模型
    model = smplx.create(
        model_path=model_path, 
        model_type='smplx', 
        gender='neutral', 
        use_pca=False, 
        batch_size=1
    ).to(device)
    model = attach_coap(model, device=device)
    
    print(f"[Pipeline] 3. Starting Frame-by-Frame Optimization ({num_frames} frames)...")
    
    # 逐帧优化循环
    for i in tqdm(range(num_frames), desc="Optimizing"):
        # 提取当前帧参数 (Slice to keep dimensions [1, ...])
        current_frame_params = {
            'global_orient': original_params['global_orient'][i:i+1],
            'body_pose':     original_params['body_pose'][i:i+1],
            'transl':        original_params['transl'][i:i+1],
            'betas':         original_params['betas'] # Constant across frames
        }
        
        # 运行优化 (这里仅优化 body_pose，其他保持)
        opt_frame = optimize_single_frame_coap(
            model, 
            current_frame_params,
            max_iters=30, # 每帧迭代次数可调
            lr=0.02,
            verbose=True # 关闭详细日志以保持进度条整洁
        )
        
        # 必须加 .cpu()，绝不要让结果留在 GPU 上！
        optimized_results['global_orient'].append(opt_frame['global_orient'].cpu())
        optimized_results['body_pose'].append(opt_frame['body_pose'].cpu())
        optimized_results['transl'].append(opt_frame['transl'].cpu())
        # betas 不需要每帧存，最后统一即可，这里为了对齐方便先存着或者最后处理
        
        torch.cuda.empty_cache()

    print("[Pipeline] 4. Aggregating Results...")
    # 将 List[Tensor] 堆叠回 Tensor [Frames, ...]
    final_params = {
        'global_orient': torch.cat(optimized_results['global_orient'], dim=0),
        'body_pose':     torch.cat(optimized_results['body_pose'], dim=0),
        'transl':        torch.cat(optimized_results['transl'], dim=0),
        'betas':         original_params['betas'] # 复用原始 betas
    }
    
    return model, final_params

def visualize_results(model, params):
    
    
    print("[Vis] Starting Result Playback using PyRender...")

    # 1. 设置场景
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[1.0, 1.0, 1.0])
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light)
    
    # 2. 启动 Viewer (先启动以避免 np.infty 问题)
    viewer = pyrender.Viewer(scene, run_in_thread=True, use_raymond_lighting=True)
    
    num_frames = params['body_pose'].shape[0]
    
    # 3. 预计算所有 Mesh (为了流畅播放，或者也可以边播边算)
    # 为了演示简单，我们在播放循环中实时计算 Mesh (Forward Pass)
    
    print("[Vis] Press 'q' to quit viewer.")
    
    while viewer.is_active:
        for i in range(num_frames):
            if not viewer.is_active: break
            
            # --- 模型前向传播获取 Mesh ---
            # 取出当前帧参数 (增加 batch 维度)
            with torch.no_grad():
                output = model(
                    global_orient=params['global_orient'][i:i+1].to(device),
                    body_pose=params['body_pose'][i:i+1].to(device),
                    transl=params['transl'][i:i+1].to(device),
                    betas=params['betas'].to(device),
                    return_verts=True
                )
                verts = output.vertices[0].cpu().numpy()
                faces = model.faces
            
            # --- 构建 PyRender Mesh ---
            trimesh_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            trimesh_mesh.visual.vertex_colors = (0.2, 0.6, 0.8, 0.8) # 蓝色表示优化后
            mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)
            
            # --- 更新场景 ---
            viewer.render_lock.acquire()
            
            # 移除旧的 mesh 节点
            # 只有 mesh 节点会被清理，保留灯光和相机
            frame_nodes = [node for node in scene.mesh_nodes]
            for node in frame_nodes:
                scene.remove_node(node)
                
            # 添加新 mesh
            scene.add(mesh)
            
            viewer.render_lock.release()
            
            time.sleep(1/30.0) # 30 FPS 控制

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_path', type=str, default='data/put_jacket.npy')
    parser.add_argument('--model_folder', type=str, default='./body_models')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 运行优化管线
    model, processed_data = run_optimization_pipeline(args.npy_path, args.model_folder, device)
    
    # 2. 可视化结果
    visualize_results(model, processed_data)
