import os
import cv2
import numpy as np

def generate_lighting_mask(height, width, center_brightness=1.5, edge_brightness=0.6, shadow_strength=0.3):
    """
    生成局部高光+阴影的光照掩膜（值域0~2.5左右）
    - 中心亮（模拟阳光直射），边缘暗（模拟周围遮挡）
    - 随机生成大小不一的暗斑（模拟树叶果实造成的阴影）
    """
    # 1. 径向渐变（中央亮，四周暗）
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    radial = 1 - (dist / max_dist) * 0.6   # 边缘亮度降至0.4倍中心
    radial = radial * (center_brightness - edge_brightness) + edge_brightness

    # 2. 随机斑点阴影（模拟树叶/果实的局部遮挡）
    np.random.seed(42)   # 固定种子可重复，也可去掉
    noise = np.random.rand(height, width)
    # 二值化产生斑点区域
    spots = (noise > 0.85).astype(np.float32)   # 约15%区域为阴影
    # 使用高斯模糊让阴影边缘柔和
    spots = cv2.GaussianBlur(spots, (15, 15), 10)
    # 阴影强度：使该区域亮度乘以 (1 - shadow_strength)
    shadow_mask = 1 - spots * shadow_strength

    # 3. 合并径向渐变和阴影斑点
    lighting_mask = radial * shadow_mask
    # 限制范围防止过曝
    lighting_mask = np.clip(lighting_mask, 0.3, 2.2)
    return lighting_mask

def adjust_light_with_color_temp(image, global_brightness, time_of_day, lighting_mask=None):
    """
    综合调整：全局曝光 + 色温 + 局部光照掩膜
    image      : BGR 格式 (H,W,3)
    global_brightness : 全局曝光系数（原 brightness 参数）
    time_of_day: 'morning', 'midday', 'evening' 影响色温
    lighting_mask: (H,W) 浮点掩膜，若为None则自动生成
    """
    # 转换为 float32 并归一化到 [0,1]
    img_float = image.astype(np.float32) / 255.0

    # 1. 局部光照掩膜（空间非均匀亮度）
    if lighting_mask is None:
        h, w = image.shape[:2]
        lighting_mask = generate_lighting_mask(h, w)
    # 掩膜扩展到三个通道
    mask_3ch = np.stack([lighting_mask, lighting_mask, lighting_mask], axis=-1)

    # 2. 应用全局亮度 + 局部掩膜
    img_bright = img_float * global_brightness * mask_3ch

    # 3. 色温模拟（改变 RGB 通道比例）
    #    morning: 偏红 (R 增益高，B 增益低)
    #    midday : 中性 (R=G=B)
    #    evening: 偏红/橙 (更红一些)
    if time_of_day == 'morning':
        # 色温较低（约3000K），偏红
        gains = np.array([1.2, 1.0, 0.7])   # R, G, B  (注意OpenCV BGR顺序：B,G,R)
    elif time_of_day == 'evening':
        # 夕阳更红
        gains = np.array([1.3, 0.9, 0.6])
    else:   # 'midday' 高色温（6500K），略微偏蓝
        gains = np.array([0.95, 1.0, 1.1])

    # 应用色温增益 (BGR顺序)
    img_color = img_bright * gains

    # 4. 增加高光区域的轻微曝光柔化（模拟过曝感）
    #    找到亮度大于阈值的区域，略微增加其饱和度或亮度
    gray = cv2.cvtColor((img_color * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255.0
    highlight_mask = (gray > 0.9).astype(np.float32)
    highlight_mask = cv2.GaussianBlur(highlight_mask, (5,5), 0)
    # 高光区域再提亮10%
    img_color = img_color * (1 + highlight_mask[:,:,np.newaxis] * 0.1)

    # 5. 裁剪并转回 uint8
    img_adjusted = np.clip(img_color * 255.0, 0, 255).astype(np.uint8)
    return img_adjusted

def process_images(input_folder, output_folder, time_param):
    """批量处理文件夹内所有图片"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    supported_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    # 可以预先为每张图生成不同的随机阴影种子，这里我们每张图重新生成掩膜（效果更丰富）
    for file in os.listdir(input_folder):
        ext = os.path.splitext(file)[1].lower()
        if ext in supported_ext:
            try:
                img_path = os.path.join(input_folder, file)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"无法读取图像：{file}")
                    continue

                # 调整图像（包含全局曝光、色温、局部光照）
                adjusted_image = adjust_light_with_color_temp(
                    image,
                    global_brightness=time_param['brightness'],
                    time_of_day=time_param['time'],
                    lighting_mask=None   # None 表示自动生成随机阴影斑点
                )

                save_path = os.path.join(output_folder, file)
                cv2.imwrite(save_path, adjusted_image)
                print(f"已处理：{file}")
            except Exception as e:
                print(f"处理失败：{file}，错误：{e}")
        else:
            print(f"跳过非图片文件：{file}")

# 设置参数（增加 time 字段用于色温选择）
time_params = {
    'morning': {'brightness': 0.7, 'time': 'morning'},
    'midday':  {'brightness': 1.1, 'time': 'midday'},
    'evening': {'brightness': 0.5, 'time': 'evening'}
}
selected_time = 'midday'   # 可改为 'morning' 或 'evening'

# 输入输出路径
input_folder = r"C:\Users\Administrator\Desktop\pythonProject\boobo\SYE"
output_folder = r"C:\Users\Administrator\Desktop\pythonProject\boobo\SYE_OUT_1.1"

# 执行批量处理
process_images(input_folder, output_folder, time_params[selected_time])