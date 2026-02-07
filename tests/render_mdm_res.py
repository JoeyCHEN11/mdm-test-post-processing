import numpy as np
import torch
import smplx
import pyrender
import trimesh
import time
import os
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

if not hasattr(np, 'infty'):
    np.infty = np.inf

# --- 2. Core Player Class ---

class MotionPlayerSMPLX:
    def __init__(self, npy_path, model_folder):
        print(f"[Info] Loading motion data: {npy_path}")
        data = np.load(npy_path, allow_pickle=True).item()
        
        # A. Data Cleaning
        raw_thetas = data['thetas'] 
        # Unify to (Frames, 24, 6)
        if len(raw_thetas.shape) == 4:
            self.thetas = torch.from_numpy(raw_thetas[0]).permute(2, 0, 1).float()
        elif len(raw_thetas.shape) == 3:
            self.thetas = torch.from_numpy(raw_thetas).permute(2, 0, 1).float()
            
        if self.thetas.shape[1] > 24:
            self.thetas = self.thetas[:, :24, :]
            
        self.trans = torch.from_numpy(data['root_translation']).permute(1, 0).float()
        self.num_frames = self.thetas.shape[0]

        # B. Initialize Model
        print(f"[Info] Loading SMPL-X model (Path: {model_folder})...")
        self.model = smplx.create(
            model_path=model_folder, 
            model_type='smplx', 
            gender='neutral', 
            use_pca=False,
            batch_size=1
        )
        
        # Pre-computation
        print("[Info] Calculating skinning...")
        self.meshes = []
        
        with torch.no_grad():
            
            # 2. Then convert to Axis-Angle (to feed into SMPL-X)
            # This step is the key to fixing "165 vs 55" error
            rot_axis_angle = matrix_to_axis_angle(rotation_6d_to_matrix(self.thetas)) # [Frames, 24, 3]
            
            for i in range(self.num_frames):
                # Split parameters
                # rot_axis_angle[i:i+1] shape is [1, 24, 3]
                
                global_orient = rot_axis_angle[i:i+1, 0:1, :] # [1, 1, 3]
                body_pose = rot_axis_angle[i:i+1, 1:22, :]    # [1, 21, 3]
                
                # Feed to model
                output = self.model(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    transl=self.trans[i:i+1],
                    betas=torch.zeros(1, 10),
                )
                
                verts = output.vertices[0].cpu().numpy()
                mesh = trimesh.Trimesh(vertices=verts, faces=self.model.faces)
                mesh.visual.vertex_colors = [180, 180, 200, 255]
                self.meshes.append(mesh)

    def run(self):
        # Setup Scene
        scene = pyrender.Scene()
        # Initialize nodes
        self.mesh_node = pyrender.Mesh.from_trimesh(self.meshes[0])
        self.node_ref = scene.add(self.mesh_node)
        
        # Lights
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=np.eye(4))
        # Fill light (prevent back being too dark)
        scene.add(pyrender.DirectionalLight(color=[0.5, 0.5, 0.5], intensity=1.0), 
                  pose=np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]]))
        
        print("[Info] Starting player (Press 'h' for help)...")
        
        # Start Viewer and capture instance
        # run_in_thread=True lets Viewer run in background, returning Viewer instance to main thread
        self.viewer = pyrender.Viewer(scene, run_in_thread=True, use_raymond_lighting=True)
        
        # Animation Loop
        print("[Info] Starting loop playback...")
        while self.viewer.is_active: # Automatically exit loop when window closes
            for i in range(self.num_frames):
                if not self.viewer.is_active:
                    break
                self.viewer.render_lock.acquire() 
                try:
                    # Remove old frame
                    scene.remove_node(self.node_ref)
                    # Add new frame
                    self.node_ref = scene.add(pyrender.Mesh.from_trimesh(self.meshes[i]))
                finally:
                    self.viewer.render_lock.release()
                
                time.sleep(1/30.0)

# --- Runtime Configuration ---
model_folder = './body_models' 
npy_file = 'data/put_jacket.npy'

if os.path.exists(npy_file):
    try:
        player = MotionPlayerSMPLX(npy_file, model_folder)
        player.run()
    except Exception as e:
        print(f"[Error] Runtime error: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"[Error] Data file not found: {npy_file}")