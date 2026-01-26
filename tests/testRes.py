import numpy as np
import os

def inspect_npy(file_path):
    if not os.path.exists(file_path):
        print(f"[Error] 文件不存在: {file_path}")
        return

    print(f"--- 正在分析: {os.path.basename(file_path)} ---")
    
    # 1. 加载数据 (MDM 项目通常需要 allow_pickle=True)
    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"[Error] 加载失败: {e}")
        return

    # 2. 判断数据类型并解包
    # 情况 A: 数据被包装在 0-d 数组中 (常见于保存 dict 的情况)
    if data.ndim == 0:
        print("[Type] 0-d Array (Wrapping a Dictionary/Object)")
        real_data = data.item() # 提取核心内容
        
        if isinstance(real_data, dict):
            print("[Content] Dictionary Keys found:")
            for key, val in real_data.items():
                # 打印每个 key 的形状，如果是数组的话
                if isinstance(val, np.ndarray):
                    print(f"  > Key: '{key}' | Shape: {val.shape} | Type: {val.dtype}")
                    # 简单统计，防止数据爆炸或全零
                    print(f"    - Range: [{val.min():.4f}, {val.max():.4f}]")
                else:
                    print(f"  > Key: '{key}' | Value: {val}")
        else:
            print(f"[Content] Object type: {type(real_data)}")
            
    # 情况 B: 数据直接就是高维数组 (常见于 raw features)
    else:
        print(f"[Type] Raw Array")
        print(f"[Shape] {data.shape}")
        print(f"[Dtype] {data.dtype}")
        print(f"[Stats] Min: {data.min():.4f}, Max: {data.max():.4f}, Mean: {data.mean():.4f}")
        
        # 针对 MDM/SMPL 的特定检查
        B, F, C = 0, 0, 0
        if len(data.shape) == 3:
            B, F, C = data.shape
            print(f"\n[推测语义 - Based on MDM context]")
            print(f"  > Batch Size (样本数): {B}")
            print(f"  > Frames (帧数): {F}")
            print(f"  > Channels (特征维度): {C}")
            
            if C == 263:
                print("  > 提示: 这看起来是 HumanML3D 的原始特征格式 (Pos+Vel+Rot)，不是纯 SMPL 参数。")
            elif C == 72 or C == 24*3:
                print("  > 提示: 这看起来是 SMPL Pose 参数 (Axis-Angle 格式)。")

# --- 使用示例 ---
# 替换为你实际的 npy 文件路径
file_path = "data/mdm_smpl_res_hug.npy" 
# 如果你还没有文件，这里生成一个模拟的 SMPL 格式文件供测试
# mock_data = {
#     'thetas': np.random.rand(1, 120, 24, 3).astype(np.float32), # (Batch, Frames, Joints, Rot)
#     'root_translation': np.random.rand(1, 120, 3).astype(np.float32),
#     'betas': np.zeros((1, 10)).astype(np.float32)
# }
# np.save("test_smpl.npy", mock_data)

# 运行分析
inspect_npy(file_path)