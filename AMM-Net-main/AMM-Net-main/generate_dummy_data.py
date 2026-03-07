import os
import pandas as pd
import numpy as np
from PIL import Image

def create_dummy_dataset():
    # 1. 创建存放假图片的文件夹
    img_dir = 'dummy_images'
    os.makedirs(img_dir, exist_ok=True)
    
    data = []
    print(f"开始生成 10 条测试数据...")
    
    for i in range(1, 11):
        image_id = f"1000{i}"
        image_name = f"{image_id}.jpg"
        image_path = os.path.join(img_dir, image_name)
        
        # 2. 生成一张 224x224 的随机纯色假图片并保存
        random_color = tuple(np.random.randint(0, 256, size=3).tolist())
        img = Image.new('RGB', (224, 224), color=random_color)
        img.save(image_path)
        
        # 3. 构造字典记录
        record = {
            'image_id': image_id,
            'image_path': image_name,
            'comment': f"This is a fake comment for image {i}. The lighting is good and colors are vivid."
        }
        
        # 4. 生成随机的 10-bin 概率分布 (保证和为 1)
        random_votes = np.random.rand(10)
        probabilities = random_votes / random_votes.sum()
        
        for j in range(1, 11):
            record[f'prob_{j}'] = probabilities[j-1]
            
        data.append(record)

    # 5. 导出为 CSV 文件
    df = pd.DataFrame(data)
    csv_filename = 'train_aligned.csv'
    df.to_csv(csv_filename, index=False)
    
    print(f"✅ 生成完毕！")
    print(f" - 图片保存在: {img_dir}/ 文件夹下 (共 10 张)")
    print(f" - 标签保存在: {csv_filename} 文件中")

if __name__ == '__main__':
    create_dummy_dataset()