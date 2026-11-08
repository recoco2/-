import cv2
import numpy as np
import os


def adjust_light(image, brightness):
    """调整图像亮度，保持颜色中性"""
    img_float = image.astype(np.float32) * brightness
    img_adjusted = np.clip(img_float, 0, 255).astype(np.uint8)
    return img_adjusted


def process_images(input_folder, output_folder, time_param):
    """批量处理文件夹内所有图片"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    supported_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    for file in os.listdir(input_folder):
        ext = os.path.splitext(file)[1].lower()
        if ext in supported_ext:
            try:
                # 读取图片
                img_path = os.path.join(input_folder, file)
                image = cv2.imread(img_path)

                # 调整亮度
                adjusted_image = adjust_light(image, time_param['brightness'])

                # 保存图片（保持原文件名）
                save_path = os.path.join(output_folder, file)
                cv2.imwrite(save_path, adjusted_image)
                print(f"已处理：{file}")
            except Exception as e:
                print(f"处理失败：{file}，错误：{e}")
        else:
            print(f"跳过非图片文件：{file}")


# 设置参数
time_params = {
    'morning': {'brightness': 0.7},
    'midday': {'brightness': 1.4},
    'evening': {'brightness': 0.5}
}
selected_time = 'midday'  # 可改为 'midday' 或 'evening'

# 输入输出路径
input_folder = r"C:\\Users\\Administrator\\Desktop\\pythonProject\\boobo\\SYE"  # 替换为你的图片文件夹路径
output_folder = r"C:\\Users\\Administrator\\Desktop\\pythonProject\\boobo\\SYE OUT1.4"  # 自动创建

# 执行批量处理
process_images(input_folder, output_folder, time_params[selected_time])
