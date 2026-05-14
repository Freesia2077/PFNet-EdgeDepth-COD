import torch
import cv2
import os
import glob
import sys

# 1. 离线加载 MiDaS 模型
model_type = "MiDaS_small" 
midas_dir = "/root/.cache/torch/hub/intel-isl_MiDaS_master"
sys.path.insert(0, midas_dir)

from midas.midas_net_custom import MidasNet_small

# 载入本地权重
model_path = "/root/.cache/torch/hub/checkpoints/midas_v21_small_256.pt"
midas = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device).eval()

# 2. 【核心修改】设置绝对路径
# 根据你 ls 的结果，确定图片在 Imgs 文件夹中
base_path = '/root/autodl-tmp/深度学习大作业/data/NEW'

data_folders = [
    os.path.join(base_path, 'train'),
    os.path.join(base_path, 'test/CHAMELEON'),
    os.path.join(base_path, 'test/CAMO'),
    os.path.join(base_path, 'test/COD10K')
]

print("开始生成深度图...")

for folder in data_folders:
    # 根据你的实际目录结构，图片目录叫 Imgs
    img_dir = os.path.join(folder, 'Imgs') 
    depth_dir = os.path.join(folder, 'depth')
    
    # 如果 Imgs 目录不存在（比如训练集还没下好），就跳过
    if not os.path.exists(img_dir):
        print(f"跳过 {folder}: 找不到 Imgs 文件夹")
        continue

    if not os.path.exists(depth_dir): 
        os.makedirs(depth_dir)
    
    # 搜索图片，兼容大小写
    img_list = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.JPG"))
    print(f"正在处理: {folder}, 发现 {len(img_list)} 张图片...")

    for img_path in img_list:
        img_name = os.path.basename(img_path).replace(".jpg", ".png").replace(".JPG", ".png")
        
        if os.path.exists(os.path.join(depth_dir, img_name)):
            continue

        img = cv2.imread(img_path)
        if img is None: continue
        img_orig_shape = img.shape[:2] # 记录原图大小 (H, W)
        
        # --- 修改部分：强制调整尺寸为 256x256 (MiDaS small 的标准尺寸) ---
        img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img_input, (256, 256)) # 强制 32 倍数
        
        input_batch = torch.unsqueeze(torch.from_numpy(img_input).permute(2, 0, 1).float() / 255.0, 0).to(device)
        
        with torch.no_grad():
            prediction = midas(input_batch)
            # 这里的尺寸要插值回原图大小 img_orig_shape
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_orig_shape, 
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        # --- 修改结束 ---

        output = prediction.cpu().numpy()
        output = (output - output.min()) / (output.max() - output.min() + 1e-5) * 255 
        cv2.imwrite(os.path.join(depth_dir, img_name), output.astype('uint8'))

print("所有深度图生成完毕！")