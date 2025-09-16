import json
import os

# 加载 COCO JSON 文件
with open('_annotations.coco.json', 'r') as f:
    data = json.load(f)

# 创建字典便于查找
images = {img['id']: img for img in data['images']}
categories = {cat['id']: cat for cat in data['categories']}
annotations = data['annotations']

output_dir = 'val'
os.makedirs(output_dir, exist_ok=True)

# 为每个图像创建对应的 txt 文件
for img_id, img in images.items():
    img_width = img['width']
    img_height = img['height']

    # 获取不带扩展名的文件名，并添加 .txt 后缀
    txt_filename = os.path.splitext(img['file_name'])[0] + '.txt'

    # 将 txt 文件保存到 test 目录下
    txt_filepath = os.path.join(output_dir, txt_filename)

    with open(txt_filepath, 'w') as txt_file:
        for ann in annotations:
            if ann['image_id'] == img_id:
                category_id = ann['category_id']
                x, y, w, h = ann['bbox']

                # 计算中心点并归一化
                center_x = (x + w / 2) / img_width
                center_y = (y + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height


                txt_file.write(f"{category_id} {center_x:.6f} {center_y:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print(f"所有标签已成功保存到 '{output_dir}' 文件夹中。")