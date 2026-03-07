import os
# 1. 优先配置网络环境，确保后续的预训练模型调用畅通无阻
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from transformers import BertModel
from torchvision import transforms

# 2. 从我们已经跑通的 Test.py 中直接导入模型和关键组件
# 这样可以避免重复写代码，保证前后端逻辑绝对一致
from Test import catNet, emd_loss, txt_process

# ==========================================
# 第一部分：定义专门处理 10-bin 审美分布的数据加载器
# ==========================================
class DummyAVADataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        
        # Swin Transformer 主分支预处理 (448x448)
        self.transform_main = transforms.Compose([
            transforms.Resize(size=(448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # CLIP 属性分支专用预处理 (必须是 224x224 且包含中心裁剪)
        self.transform_clip = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 1. 读取并处理图像
        img_path = os.path.join(self.img_dir, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        
        img_main = self.transform_main(image)
        img_clip = self.transform_clip(image)
        
        # 2. 处理文本评论
        text_comment = str(row['comment'])
        txt_tensor = txt_process(text_comment) # 调用 Test.py 中的 padding 函数
        
        # 3. 提取 1-10 分的概率分布标签
        prob_cols = [f'prob_{i}' for i in range(1, 11)]
        label_tensor = torch.tensor(row[prob_cols].values.astype('float32'))
        
        return img_main, txt_tensor, img_clip, label_tensor

# ==========================================
# 第二部分：核心训练闭环
# ==========================================
def train_loop():
    # 检测是否可以使用您强大的 RTX 显卡
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # "cuda" if torch.cuda.is_available() else 
    print(f"🚀 正在使用 {device} 设备进行训练测试...")

    print("📦 正在加载 BERT 与您重构的 AMM-Net 模型...")
    bert = BertModel.from_pretrained('bert-base-uncased')
    model = catNet(bert).to(device)
    
    # 强制进入训练模式 (这会让 Dropout 和 BatchNorm 开始发挥作用)
    model.train() 

    # 初始化数据集与 Dataloader
    dataset = DummyAVADataset(csv_file='train_aligned.csv', img_dir='dummy_images')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 定义损失函数与优化器
    criterion = emd_loss(dist_r=1) # 采用原论文的 EMD 损失来衡量分布差异
    # 过滤出所有 requires_grad=True 的参数进行优化
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    epochs = 5
    print("\n🔥 开始反向传播点火测试...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (img_main, txt, img_clip, label) in enumerate(dataloader):
            # 将批次数据转移到 GPU/CPU
            img_main = img_main.to(device)
            txt = txt.to(device)
            img_clip = img_clip.to(device)
            label = label.to(device)
            
            # --- 核心四步曲 ---
            optimizer.zero_grad()                  # 1. 梯度清零
            output = model(img_main, txt, img_clip)# 2. 前向传播
            loss = criterion(output, label)        # 3. 计算 EMD 误差
            loss.backward()                        # 4. 反向传播 (算梯度)
            optimizer.step()                       # 5. 参数更新
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | 平均 EMD Loss: {avg_loss:.4f}")
        
    print("\n✅ 闭环测试结束！")

if __name__ == '__main__':
    train_loop()