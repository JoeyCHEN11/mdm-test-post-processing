import torch
import smplx
import pyrender
import trimesh
import time
import numpy as np
import torch.nn.functional as F
import argparse

# 导入你的数据读取模块
from mdm_read import load_mdm_to_smplx_params
# 导入 COAP (确保 coap 包在 python path 下)
from coap import attach_coap

def visualize_frame(viewer, model, density_samples=None):
    """ 更新 viewer 场景中的 mesh 和点云 """
    if viewer is None: return

    # 获取 viewer 锁
    viewer.render_lock.acquire()
    
    # 1. 清理旧节点
    # 注意：不要清理 light 或 camera，只清理 mesh
    bg_nodes = [node for node in viewer.scene.nodes if isinstance(node, pyrender.Node) and (node.mesh is None or node.light is not None or node.camera is not None)]
    # 这里简单起见，我们假设 viewer.scene.mesh_nodes 存的是我们要删的
    while len(viewer.scene.mesh_nodes) > 0:
        viewer.scene.mesh_nodes.pop()
        
    # 2. 从当前 SMPL 模型提取最新的 mesh
    # 我们假设 model 刚刚做完 forward，vertices 已经是新的了
    # 但实际上我们需要重新 forward 一次或者访问缓存，为了保险，我们在外部 forward 后把 vertices 传进来
    # 这里为了代码解耦，我们传入 model，并在内部假定 model 状态是最新的
    # *更好的做法是直接传入 verts*，不过为了简单，我们重新取一次 output (不带 grad)
    # 此处假设外部已经做了 forward，我们只取 vertices (注意: 这里有性能开销，测试用无所谓)
    
    # 获取 Vertices (Detach grad)
    if hasattr(model, 'last_vertices'):
        verts = model.last_vertices.detach().cpu().numpy()[0]
    else:
        # Fallback if attribute not found
        return 

    # 创建 Mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=model.faces)
    mesh.visual.vertex_colors = (0.7, 0.7, 0.7, 0.8) # 灰色半透明
    viewer.scene.add(pyrender.Mesh.from_trimesh(mesh))

    # 3. 渲染碰撞检测点 (红球)
    if density_samples is not None:
        pts = density_samples.detach().cpu().numpy() # [N, 3]
        if pts.shape[0] > 0:
            # 实例化渲染大量小球 (Instance Rendering)
            sm = trimesh.creation.uv_sphere(radius=0.01)
            sm.visual.vertex_colors = (1.0, 0.0, 0.0) # 红色
            
            tfs = np.tile(np.eye(4), (pts.shape[0], 1, 1))
            tfs[:, :3, 3] = pts
            viewer.scene.add(pyrender.Mesh.from_trimesh(sm, poses=tfs))

    viewer.render_lock.release()


def main():
    # ---------------- 1. 配置参数 ----------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_folder = './body_models'
    data_path = 'data/mdm_smpl_res_hug.npy'
    
    # COAP 优化参数
    lr = 0.01
    max_iters = 100
    selfpen_weight = 10.0 # 提高一点权重让效果更明显
    pose_prior_weight = 100.0 # 必须要有约束，否则人会飞走或扭曲
    frame_num = 17

    print(f"[Init] 运行设备: {device}")

    # ---------------- 2. 准备模型 (COAP Ready) ----------------
    print("[Init] 加载 SMPL-X 并挂载 COAP...")
    # 注意: COAP 需要 smplx 模型
    model = smplx.create(
        model_path=model_folder, 
        model_type='smplx', 
        gender='neutral', 
        use_pca=False, 
        batch_size=1
    ).to(device)
    
    # 挂载 COAP 模块
    # 注意: attach_coap 会修改 model，注入 geometry cache 等
    model = attach_coap(model, device=device)

    # ---------------- 3. 准备数据 (单帧) ----------------
    full_data = load_mdm_to_smplx_params(data_path, device=device)
    
    # 提取第 0 帧并增加 Batch 维度 [1, ...]
    init_data = {
        'global_orient': full_data['global_orient'][frame_num:frame_num + 1].detach().clone(),
        'body_pose':     full_data['body_pose'][frame_num:frame_num + 1].detach().clone(),
        'transl':        full_data['transl'][frame_num:frame_num + 1].detach().clone(),
        'betas':         full_data['betas'].detach().clone()
    }

    # 将需要优化的参数设为可导
    # 我们通常优化 body_pose，global_orient 和 transl 一般不需要为了自穿模而大幅修改
    # 但为了防止某些极端情况，可以微调 global_orient
    init_data['body_pose'].requires_grad = True
    
    # 记录初始姿态用于 Prior Loss (防止动作走样)
    initial_pose_ref = init_data['body_pose'].detach().clone()

    # 优化器
    optimizer = torch.optim.Adam([init_data['body_pose']], lr=lr)

    # ---------------- 4. 渲染器启动 ----------------
    print("[Vis] 启动渲染器...")
    scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
    # 添加直射光
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=np.eye(4))
    
    viewer = pyrender.Viewer(scene, run_in_thread=True, use_raymond_lighting=True)
    
    # ---------------- 5. 优化循环 ----------------
    print("[Opt] 开始优化循环...")
    
    for i in range(max_iters):
        optimizer.zero_grad()
        
        # A. 前向传播
        # COAP 修改了 smplx 的 forward 机制或者我们需要在这里手动触发
        # 通常直接调用 model(...) 即可，COAP 的魔法在 loss 计算里
        output = model(
            global_orient=init_data['global_orient'],
            body_pose=init_data['body_pose'],
            transl=init_data['transl'],
            betas=init_data['betas'],
            return_verts=True,
            return_full_pose=True # 必须，用于 COAP 内部计算
        )
        
        # Hack: 将当前 verts 存入 model 方便 visualize 函数读取
        model.last_vertices = output.vertices
        
        # B. 计算 Loss
        # 1. 自穿模 Loss (COAP 核心)
        # ret_samples=True 会返回穿模的点，用于可视化红色小球
        loss_sp, collision_samples = model.coap.self_collision_loss(output, ret_samples=True)
        
        # 2. 姿态保持 Loss (MSE)
        # 确保动作不要为了躲避穿模而变得完全不像原来的动作
        loss_prior = F.mse_loss(init_data['body_pose'], initial_pose_ref)
        
        total_loss = (loss_sp * selfpen_weight) + (loss_prior * pose_prior_weight)
        
        # C. 反向传播
        total_loss.backward()
        optimizer.step()
        
        # D. 实时可视化更新
        # 提取第一个样本的碰撞点 [1, N, 3] -> [N, 3]
        samples_vis = collision_samples[0] if collision_samples is not None and len(collision_samples)>0 else None
        
        visualize_frame(viewer, model, samples_vis)
        
        print(f"Iter {i:03d} | Total: {total_loss.item():.4f} | SP: {loss_sp.item():.4f} | Prior: {loss_prior.item():.4f}")
        
        if loss_sp.item() < 0.1: # 随便设个阈值作为早停
            print("Converged! (No severe collision)")
            # break # 为了看演示效果，可以不 break，让它继续稳定一会
            
        time.sleep(0.05) # 减慢一点速度方便人眼观察

    print("Done. Closing in 5 seconds.")
    time.sleep(5)
    viewer.close_external()

if __name__ == "__main__":
    main()