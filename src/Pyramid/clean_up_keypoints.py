import numpy as np
import matplotlib.pyplot as plt
from functools import cmp_to_key

def compare_keypoints(kp1, kp2):
    """
    比较两个关键点的优先级（用于排序）
    
    比较规则（优先级从高到低）:
    1. x坐标（升序）
    2. y坐标（升序）
    3. 特征尺寸（降序，大尺寸优先）
    4. 方向角度（升序）
    5. 响应值（降序，高响应优先）
    6. 组/层信息（降序，高层优先）
    7. 类别ID（降序，高ID优先）
    
    返回:
    int: 负数表示kp1应排在kp2前，正数反之，0表示相等
    """
    # 1. 比较x坐标
    if kp1['x'] != kp2['x']:
        return 1 if kp1['x'] > kp2['x'] else -1
    
    # 2. 比较y坐标
    if kp1['y'] != kp2['y']:
        return 1 if kp1['y'] > kp2['y'] else -1
    
    # 3. 比较特征尺寸（降序）
    if kp1['size'] != kp2['size']:
        return -1 if kp1['size'] > kp2['size'] else 1
    
    # 4. 比较方向角度（升序）
    if kp1.get('orientation', 0) != kp2.get('orientation', 0):
        return 1 if kp1['orientation'] > kp2['orientation'] else -1
    
    # 5. 比较响应值（降序）
    if kp1['response'] != kp2['response']:
        return -1 if kp1['response'] > kp2['response'] else 1
    
    # 6. 比较组/层信息（降序）
    if kp1['octave'] != kp2['octave']:
        return -1 if kp1['octave'] > kp2['octave'] else 1
    
    # 7. 比较类别ID（降序）
    return -1 if kp1.get('class_id', -1) > kp2.get('class_id', -1) else 1

def remove_duplicate_keypoints(keypoints, tol=1e-7):
    """
    对关键点排序并移除重复的关键点
    
    参数:
    keypoints (list): 关键点字典列表
    tol (float): 浮点数比较容差（默认1e-7）
    
    返回:
    list: 去重后的关键点列表
    
    去重条件:
    当两个关键点的以下属性在容差范围内相等时视为重复:
    - 位置 (x, y)
    - 尺寸 (size)
    - 方向 (orientation)
    """
    # 处理边界情况
    if len(keypoints) < 2:
        return keypoints
    
    # 使用比较函数排序
    sorted_keypoints = sorted(keypoints, key=cmp_to_key(compare_keypoints))
    
    # 初始化结果列表
    unique_keypoints = [sorted_keypoints[0]]
    
    # 遍历排序后的关键点
    for next_kp in sorted_keypoints[1:]:
        last_kp = unique_keypoints[-1]
        
        # 检查是否为重复关键点
        is_duplicate = (
            abs(last_kp['x'] - next_kp['x']) < tol and
            abs(last_kp['y'] - next_kp['y']) < tol and
            abs(last_kp['size'] - next_kp['size']) < tol and
            abs(last_kp.get('orientation', 0) - next_kp.get('orientation', 0)) < tol
        )
        
        # 非重复关键点则添加到结果
        if not is_duplicate:
            unique_keypoints.append(next_kp)
    
    return unique_keypoints

def convert_keypoints_to_input_image_size(keypoints):
    """将关键点坐标、尺寸和八度信息转换到输入图像尺寸
    
    参数:
    keypoints (list): 关键点字典列表
    
    返回:
    list: 转换后的关键点列表
    
    转换规则:
    1. 坐标 (x, y) 乘以 0.5
    2. 尺寸 (size) 乘以 0.5
    3. 八度信息 (octave) 减1 (保持高位不变)
    """
    converted_keypoints = []
    for keypoint in keypoints:
        # 创建副本避免修改原始数据
        kp = keypoint.copy()
        # 坐标缩小一半
        kp['x'] *= 0.5
        kp['y'] *= 0.5
        # 尺寸缩小一半
        kp['size'] *= 0.5
        # 调整八度信息
        octave = kp['octave']
        kp['octave'] = (octave & ~255) | ((octave - 1) & 255)
        converted_keypoints.append(kp)
    return converted_keypoints

def visualize_converted_keypoints(image, keypoints, title="Converted and cleaned keypoints to input image size"):
    """
    在图像上可视化已经转换到输入图像尺寸的关键点
    
    参数:
    image (np.array): 原始图像（输入图像）
    keypoints (list): 已经通过convert_keypoints_to_input_image_size转换后的关键点列表
    title (str): 图像标题
    
    与原始visualize_keypoints的区别:
    1. 标题明确表示显示的是转换后的关键点
    2. 不需要调整坐标，因为关键点已经是原始图像尺寸
    3. 尺寸显示更准确，因为已经是原始图像尺寸
    """
    # 转换为RGB以便绘制彩色圆圈
    img_array = np.array(image)

    if len(img_array.shape) == 2:  # 如果是灰度图
        img_rgb = np.stack([img_array, img_array, img_array], axis=2)
    else:
        img_rgb = img_array
    
    # 获取图像尺寸
    height, width = img_array.shape[:2]
        
    # 创建图像
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb, cmap='gray')
    
    # 绘制关键点
    for kp in keypoints:
        # 关键点坐标已经是原始图像尺寸
        x, y = kp['x'], kp['y']
        
        # 确保坐标在图像范围内
        if 0 <= x < width and 0 <= y < height:
            # 使用原始尺寸显示（不需要调整）
            size = kp['size'] / 2  # 圆的半径
            
            # 绘制圆圈表示关键点位置和大小
            circle = plt.Circle((x, y), size, color='g', fill=False, linewidth=1.5)
            plt.gca().add_patch(circle)
            
            # 绘制关键点中心（使用绿色区分）
            plt.plot(x, y, 'g.', markersize=3)
    
    plt.title(f"{title} - {len(keypoints)} keypoints")
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=True)





