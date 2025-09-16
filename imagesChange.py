# # 双边滤波
# import cv2
# import os
# from glob import glob
#
# def process_images_with_bilateral_filter(input_folder, output_folder, d=9, sigma_color=75, sigma_space=75):
#
#     # 创建输出文件夹
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # 支持的图像格式
#     supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
#
#     # 获取所有图像文件
#     image_paths = glob(os.path.join(input_folder, '*'))
#     image_paths = [path for path in image_paths if os.path.isfile(path) and path.lower().endswith(supported_formats)]
#
#     if not image_paths:
#         print(f"在 {input_folder} 中未找到支持的图像文件。")
#         return
#
#     print(f"找到 {len(image_paths)} 张图像，开始处理...")
#
#     for img_path in image_paths:
#         # 读取图像
#         image = cv2.imread(img_path)
#         if image is None:
#             print(f"无法读取图像：{img_path}")
#             continue
#
#         # 双边滤波
#         filtered_image = cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
#
#         # 构造输出路径
#         filename = os.path.basename(img_path)
#         name, ext = os.path.splitext(filename)
#         output_path = os.path.join(output_folder, f"{name}{ext}")
#
#         # 保存图像
#         success = cv2.imwrite(output_path, filtered_image)
#         if success:
#             print(f"已保存：{output_path}")
#         else:
#             print(f"保存失败：{output_path}")
#
#     print("处理完成！")
#
#
#
# input_dir = r'D:/python/pythonProject/BeeDetection/BeeData/val/images'   # 输入文件夹
# output_dir = r'D:/python/pythonProject/BeeDetection/images/val'          # 输出文件夹
#
# # 执行处理
# process_images_with_bilateral_filter(input_dir, output_dir, d=9, sigma_color=75, sigma_space=75)
#



# 锐化
import cv2
import os
from glob import glob
import numpy as np


def sharpen_images(input_folder, output_folder):

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 支持的图像格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # 获取所有图像文件
    image_paths = glob(os.path.join(input_folder, '*'))
    image_paths = [path for path in image_paths
                   if os.path.isfile(path) and path.lower().endswith(supported_formats)]

    if not image_paths:
        print(f"在 {input_folder} 中未找到支持的图像文件。")
        return

    print(f"找到 {len(image_paths)} 张图像，开始锐化处理...")

    for img_path in image_paths:
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法读取图像：{img_path}")
            continue

        # 转为浮点数类型以避免溢出问题
        image_float32 = np.float32(image)

        # 使用拉普拉斯滤波器计算边缘
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # 定义锐化核
        sharpened_image = cv2.filter2D(image_float32, -1, kernel)

        # 将像素值转换回 uint8 类型
        sharpened_image = np.clip(sharpened_image, 0, 255)
        sharpened_image = np.uint8(sharpened_image)

        # 构造输出路径
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{name}{ext}")

        # 保存锐化后的图像
        success = cv2.imwrite(output_path, sharpened_image)
        if success:
            print(f"已保存锐化后的图像：{output_path}")
        else:
            print(f"保存失败：{output_path}")

    print("锐化处理完成！")



input_dir = r'D:/python/pythonProject/BeeDetection/BeeData/test/images'  # 输入文件夹
output_dir = r'D:/python/pythonProject/BeeDetection/images/test'  # 输出文件夹（建议换个名字）

# 执行锐化处理
sharpen_images(input_dir, output_dir)