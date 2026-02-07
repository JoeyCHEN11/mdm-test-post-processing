import torch
import smplx
import pyrender
import trimesh
import time
import numpy as np
import torch.nn.functional as F
import argparse

from mdm_read import load_mdm_to_smplx_params
from coap import attach_coap

def visualize_frame(viewer, model, density_samples=None):
    """ Update mesh and point cloud in viewer scene """
    if viewer is None: return

    viewer.render_lock.acquire()
    
    # Clean up old nodes
    while len(viewer.scene.mesh_nodes) > 0:
        viewer.scene.mesh_nodes.pop()
        
    # Get Vertices (Detach grad)
    if hasattr(model, 'last_vertices'):
        verts = model.last_vertices.detach().cpu().numpy()[0]
    else:
        # Fallback if attribute not found
        return 

    # Create Mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=model.faces)
    mesh.visual.vertex_colors = (0.7, 0.7, 0.7, 0.8) # Gray semi-transparent
    viewer.scene.add(pyrender.Mesh.from_trimesh(mesh))

    # Render collision detection points (red spheres)
    if density_samples is not None:
        pts = density_samples.detach().cpu().numpy() # [N, 3]
        if pts.shape[0] > 0:
            # Instantiate render large number of small spheres (Instance Rendering)
            sm = trimesh.creation.uv_sphere(radius=0.01)
            sm.visual.vertex_colors = (1.0, 0.0, 0.0) # Red
            
            tfs = np.tile(np.eye(4), (pts.shape[0], 1, 1))
            tfs[:, :3, 3] = pts
            viewer.scene.add(pyrender.Mesh.from_trimesh(sm, poses=tfs))

    viewer.render_lock.release()


def main():
    # Configure parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_folder = './body_models'
    data_path = 'data/mdm_smpl_res_hug.npy'
    
    # COAP optimization parameters
    lr = 0.01
    max_iters = 100
    selfpen_weight = 10.0 # Increase weight slightly to make effect more obvious
    pose_prior_weight = 100.0 # constraints
    frame_num = 17

    print(f"[Init] Running device: {device}")

    # Prepare smplx 
    print("[Init] Loading SMPL-X and attaching COAP...")
    model = smplx.create(
        model_path=model_folder, 
        model_type='smplx', 
        gender='neutral', 
        use_pca=False, 
        batch_size=1
    ).to(device)
    
    # Attach COAP module
    model = attach_coap(model, device=device)

    # ---------------- 3. Prepare data (Single frame) ----------------
    full_data = load_mdm_to_smplx_params(data_path, device=device)
    
    # Extract frame 0 (or specific frame) and add Batch dimension [1, ...]
    init_data = {
        'global_orient': full_data['global_orient'][frame_num:frame_num + 1].detach().clone(),
        'body_pose':     full_data['body_pose'][frame_num:frame_num + 1].detach().clone(),
        'transl':        full_data['transl'][frame_num:frame_num + 1].detach().clone(),
        'betas':         full_data['betas'].detach().clone()
    }

    # Set parameters to optimize to be differentiable
    # We usually optimize body_pose, global_orient and transl generally don't need large modifications for self-penetration
    # But to prevent extreme cases, global_orient can be fine-tuned
    init_data['body_pose'].requires_grad = True
    
    # Record initial pose for Prior Loss (prevent pose drift)
    initial_pose_ref = init_data['body_pose'].detach().clone()

    # Optimizer
    optimizer = torch.optim.Adam([init_data['body_pose']], lr=lr)

    # ---------------- 4. Start Renderer ----------------
    print("[Vis] Starting renderer...")
    scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
    # Add directional light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=np.eye(4))
    
    viewer = pyrender.Viewer(scene, run_in_thread=True, use_raymond_lighting=True)
    
    # ---------------- 5. Optimization Loop ----------------
    print("[Opt] Starting optimization loop...")
    
    for i in range(max_iters):
        optimizer.zero_grad()

        output = model(
            global_orient=init_data['global_orient'],
            body_pose=init_data['body_pose'],
            transl=init_data['transl'],
            betas=init_data['betas'],
            return_verts=True,
            return_full_pose=True # Required, for COAP internal calculation
        )

        model.last_vertices = output.vertices
        
        # Calculate Loss
        # ret_samples=True returns penetration points for visualizing red spheres
        loss_sp, collision_samples = model.coap.self_collision_loss(output, ret_samples=True)
        

        loss_prior = F.mse_loss(init_data['body_pose'], initial_pose_ref)
        
        total_loss = (loss_sp * selfpen_weight) + (loss_prior * pose_prior_weight)

        total_loss.backward()
        optimizer.step()

        samples_vis = collision_samples[0] if collision_samples is not None and len(collision_samples)>0 else None
        
        visualize_frame(viewer, model, samples_vis)
        
        print(f"Iter {i:03d} | Total: {total_loss.item():.4f} | SP: {loss_sp.item():.4f} | Prior: {loss_prior.item():.4f}")
        
        if loss_sp.item() < 0.1: # Set arbitrary threshold for early stopping
            print("Converged! (No severe collision)")
            
        time.sleep(0.05)

    print("Done. Closing in 5 seconds.")
    time.sleep(5)
    viewer.close_external()

if __name__ == "__main__":
    main()