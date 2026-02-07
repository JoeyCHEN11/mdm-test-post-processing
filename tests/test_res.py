import numpy as np
import os

def inspect_npy(file_path):
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return

    print(f"--- Analyzing: {os.path.basename(file_path)} ---")
    
    # Load data
    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"[Error] Load failed: {e}")
        return

    # Check data type
    if data.ndim == 0:
        print("[Type] 0-d Array (Wrapping a Dictionary/Object)")
        real_data = data.item() # Extract content
        
        if isinstance(real_data, dict):
            print("[Content] Dictionary Keys found:")
            for key, val in real_data.items():
                # Print shape for each key if it is an array
                if isinstance(val, np.ndarray):
                    print(f"  > Key: '{key}' | Shape: {val.shape} | Type: {val.dtype}")
                    # Simple stats
                    print(f"    - Range: [{val.min():.4f}, {val.max():.4f}]")
                else:
                    print(f"  > Key: '{key}' | Value: {val}")
        else:
            print(f"[Content] Object type: {type(real_data)}")
            
    # High-dimensional array (common for raw features)
    else:
        print(f"[Type] Raw Array")
        print(f"[Shape] {data.shape}")
        print(f"[Dtype] {data.dtype}")
        print(f"[Stats] Min: {data.min():.4f}, Max: {data.max():.4f}, Mean: {data.mean():.4f}")
        
        # Specific check for MDM/SMPL
        B, F, C = 0, 0, 0
        if len(data.shape) == 3:
            B, F, C = data.shape
            print(f"\n[Inferred Semantics - Based on MDM context]")
            print(f"  > Batch Size: {B}")
            print(f"  > Frames: {F}")
            print(f"  > Channels (Feature Dim): {C}")
            
            if C == 263:
                print("  > Hint: Looks like HumanML3D raw feature format (Pos+Vel+Rot), not pure SMPL parameters.")
            elif C == 72 or C == 24*3:
                print("  > Hint: Looks like SMPL Pose parameters (Axis-Angle format).")


file_path = "data/mdm_smpl_res_hug.npy" 
inspect_npy(file_path)