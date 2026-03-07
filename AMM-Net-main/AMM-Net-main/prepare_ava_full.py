import os
import pandas as pd
import json
from tqdm import tqdm

def prepare_full_dataset(ava_txt_path, comments_json_path, images_dir):
    print("📂 开始读取 AVA 评分数据...")
    # AVA.txt 格式: index image_id score1 score2 ... score10 tag1 tag2 ...
    ava_data = pd.read_csv(ava_txt_path, sep=' ', header=None)
    
    # 提取 ID 和 10 个评分列 (第2列是ID，第3-12列是评分)
    scores = ava_data.iloc[:, 2:12].values
    # 归一化为概率分布 (和为1)
    sums = scores.sum(axis=1, keepdims=True)
    probs = scores / sums
    
    df = pd.DataFrame({
        'image_id': ava_data.iloc[:, 1].astype(str),
    })
    for i in range(10):
        df[f'prob_{i+1}'] = probs[:, i]

    print("💬 开始匹配评论数据...")
    with open(comments_json_path, 'r', encoding='utf-8') as f:
        comments_dict = json.load(f) # 假设格式为 {"image_id": ["comment1", "comment2"]}
    
    # 提取每个 ID 的第一条评论，如果没有评论则填入默认值
    def get_first_comment(img_id):
        c_list = comments_dict.get(img_id, [])
        return c_list[0] if c_list else "This image has no description."

    df['comment'] = df['image_id'].apply(get_first_comment)
    
    # 构造完整图片路径并检查是否存在
    df['image_path'] = df['image_id'] + ".jpg"
    
    print(f"📊 原始数据共 {len(df)} 条，正在过滤损坏/缺失图片...")
    # 这一步在处理 30GB 数据时很重要，避免训练中断
    valid_mask = [os.path.exists(os.path.join(images_dir, p)) for p in tqdm(df['image_path'])]
    df = df[valid_mask]
    
    # 划分训练集和验证集 (9:1)
    train_df = df.sample(frac=0.9, random_state=42)
    val_df = df.drop(train_df.index)
    
    train_df.to_csv('train_total.csv', index=False)
    val_df.to_csv('val_total.csv', index=False)
    
    print(f"✅ 完成！训练集: {len(train_df)} 条, 验证集: {len(val_df)} 条")
    print("📝 已生成 train_total.csv 和 val_total.csv")

# ==========================================
# 请在此处填入您的真实路径
# ==========================================
# prepare_full_dataset(
#     ava_txt_path='AVA.txt', 
#     comments_json_path='AVA_Comments.json', 
#     images_dir='C:/AVA_Dataset/images'
# )