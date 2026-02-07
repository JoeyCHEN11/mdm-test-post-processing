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

# Fix for NumPy compatibility
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
    Perform COAP optimization on single-frame SMPL parameters to remove self-intersections.
    
    Args:
        model: SMPL-X model instance with attached COAP
        init_params (dict): Dictionary of Tensors containing 'global_orient', 'body_pose', 'transl', 'betas'
                            Expected shape is (1, ...) i.e. batch size is 1
        max_iters (int): Maximum optimization steps
        lr (float): Learning rate
        selfpen_weight (float): Self-penetration penalty weight
        pose_prior_weight (float): Weight to maintain original pose
        stop_threshold (float): Early stopping threshold, stops when loss_sp is less than this value
        verbose (bool): Whether to print the optimization process
        
    Returns:
        dict: Dictionary containing optimized parameters (detached tensors)
    """
    
    # Clone and set parameters
    # fine-tune body_pose to resolve intersections, keeping global_orient (orientation) and transl (translation) unchanged
    body_pose = init_params['body_pose'].clone().detach().requires_grad_(True)
    
    # Other parameters remain fixed
    global_orient = init_params['global_orient'].clone().detach()
    transl = init_params['transl'].clone().detach()
    betas = init_params['betas'].clone().detach()
    
    # Original pose reference (used to calculate deviation penalty)
    ref_body_pose = init_params['body_pose'].clone().detach()
    
    # 2. Configure optimizer
    optimizer = torch.optim.Adam([body_pose], lr=lr)
    
    # 3. Optimization loop
    for i in range(max_iters):
        optimizer.zero_grad()
        
        # SMPL Forward
        output = model(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            betas=betas,
            return_verts=True,
            return_full_pose=True
        )
        
        # Calculate Intersection Loss
        loss_sp = model.coap.self_collision_loss(output)
        
        # Early stop
        if loss_sp.item() < stop_threshold: 
            if verbose: print(f"    [Frame Opt] Early stop at iter {i}, loss_sp: {loss_sp.item():.5f}")
            break
            
        # Regularization
        loss_prior = F.mse_loss(body_pose, ref_body_pose)
        
        # Total Loss
        loss = (loss_sp * selfpen_weight) + (loss_prior * pose_prior_weight)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        if verbose:
            print(f"    [Frame Opt] Iter {i}: Total={loss.item():.4f} (SP={loss_sp.item():.4f})")
            
    optimized_params = {
        'global_orient': global_orient.detach(),
        'body_pose': body_pose.detach(),
        'transl': transl.detach(),
        'betas': betas.detach()
    }

    return optimized_params

def run_optimization_pipeline(npy_path, model_path, device='cuda'):
    print(f"[Pipeline] 1. Loading Data from {npy_path}...")
    # Load original data
    original_params = load_mdm_to_smplx_params(npy_path, device=device)
    num_frames = original_params['body_pose'].shape[0]
    
    print(f"[Pipeline] 2. Initializing SMPL-X with COAP...")
    
    # Prepare container to store optimized frames
    optimized_results = {
        'global_orient': [],
        'body_pose': [],
        'transl': [],
        'betas': []
    }
    # Initialize model
    model = smplx.create(
        model_path=model_path, 
        model_type='smplx', 
        gender='neutral', 
        use_pca=False, 
        batch_size=1
    ).to(device)
    model = attach_coap(model, device=device)
    
    print(f"[Pipeline] 3. Starting Frame-by-Frame Optimization ({num_frames} frames)...")
    
    # Frame-by-frame optimization loop
    for i in tqdm(range(num_frames), desc="Optimizing"):
        # Extract current frame parameters (Slice to keep dimensions [1, ...])
        current_frame_params = {
            'global_orient': original_params['global_orient'][i:i+1],
            'body_pose':     original_params['body_pose'][i:i+1],
            'transl':        original_params['transl'][i:i+1],
            'betas':         original_params['betas'] # Constant across frames
        }
        
        # Run optimization (here only optimizing body_pose, keeping others fixed)
        opt_frame = optimize_single_frame_coap(
            model, 
            current_frame_params,
            max_iters=30, # Adjustable iterations per frame
            lr=0.02,
            verbose=True # Turn off verbose logs to keep progress bar clean
        )
        
        # Must add .cpu(), never leave results on GPU!
        optimized_results['global_orient'].append(opt_frame['global_orient'].cpu())
        optimized_results['body_pose'].append(opt_frame['body_pose'].cpu())
        optimized_results['transl'].append(opt_frame['transl'].cpu())
        # betas don't need to be stored every frame, unified at the end, stored here for convenience alignment or processed at the end
        
        torch.cuda.empty_cache()

    print("[Pipeline] 4. Aggregating Results...")
    # Stack List[Tensor] back to Tensor [Frames, ...]
    final_params = {
        'global_orient': torch.cat(optimized_results['global_orient'], dim=0),
        'body_pose':     torch.cat(optimized_results['body_pose'], dim=0),
        'transl':        torch.cat(optimized_results['transl'], dim=0),
        'betas':         original_params['betas'] # Reuse original betas
    }
    
    return model, final_params

def visualize_results(model, params):
    
    
    print("[Vis] Starting Result Playback using PyRender...")

    # Setup scene
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[1.0, 1.0, 1.0])
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light)
    
    # Start Viewer (Start first to avoid np.infty issue)
    viewer = pyrender.Viewer(scene, run_in_thread=True, use_raymond_lighting=True)
    
    num_frames = params['body_pose'].shape[0]
    
    # Precompute all Meshes (for smooth playback, or compute on the fly)
    print("[Vis] Press 'q' to quit viewer.")
    
    while viewer.is_active:
        for i in range(num_frames):
            if not viewer.is_active: break
            
            # --- Model forward pass to get Mesh ---
            # Extract current frame parameters (add batch dimension)
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
            
            # --- Build PyRender Mesh ---
            trimesh_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            trimesh_mesh.visual.vertex_colors = (0.2, 0.6, 0.8, 0.8) # Blue indicates optimized
            mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)
            
            # --- Update scene ---
            viewer.render_lock.acquire()
            
            # Remove old mesh nodes
            # Only mesh nodes will be cleaned, lights and camera retained
            frame_nodes = [node for node in scene.mesh_nodes]
            for node in frame_nodes:
                scene.remove_node(node)
                
            # Add new mesh
            scene.add(mesh)
            
            viewer.render_lock.release()
            
            time.sleep(1/30.0) # 30 FPS control

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_path', type=str, default='data/eagle_pos.npy')
    parser.add_argument('--model_folder', type=str, default='./body_models')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    model, processed_data = run_optimization_pipeline(args.npy_path, args.model_folder, device)

    visualize_results(model, processed_data)
