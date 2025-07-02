import numpy as np
import math
    

def unpack_octave(keypoint):
    """从关键点中解包组索引、层索引和尺度信息
    
    参数:
        keypoint: 关键点字典，包含'octave'属性
        
    返回:
        group_index: 组索引（金字塔中的组）
        layer_index: 层索引（组内的尺度层）
        scale: 关键点相对于原始图像的尺度（缩放因子）
    
    关键说明：
    - 关键点的'octave'属性存储了组索引和层索引
    - 组索引(group_index): 表示关键点所在的图像金字塔组
        - 0: 原始尺寸组（未降采样）
        - 1: 降采样一次后的组（尺寸减半）
        - 2: 降采样两次后的组（尺寸为原始的1/4）
    - 层索引(layer_index): 表示在组内的尺度层
    - 尺度(scale): 关键点相对于原始图像的实际大小比例
    """
    # 1. 提取组索引（低8位）
    group_index = keypoint['octave'] & 255
    
    # 2. 提取层索引（高8位）
    layer_index = (keypoint['octave'] >> 8) & 255
    
    # 3. 处理负组索引（当组索引>=128时，表示负数）
    if group_index >= 128:
        group_index = group_index | -128
    
    # 4. 计算尺度因子
    if group_index >= 0:
        # 非负组索引：尺度 = 1 / (2^组索引)
        scale = 1.0 / (2.0 ** group_index)
    else:
        # 负组索引：尺度 = 2^(-组索引)
        scale = 2.0 ** (-group_index)
    
    return group_index, layer_index, scale

def generate_descriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    """Generate descriptors for each keypoint without using OpenCV"""   
    descriptors = []
    float_tolerance = 1e-7  # 浮点数容差

    for keypoint in keypoints:
        octave, layer, scale = unpack_octave(keypoint)  # 使用已定义的unpack_octave函数
        gaussian_image = gaussian_images[octave + 1][layer]
        if hasattr(gaussian_image, 'size'):
            num_cols, num_rows = gaussian_image.size
        else:
            num_rows, num_cols = gaussian_image.shape
        point = np.array([keypoint['x'], keypoint['y']])
        point = np.round(scale * point).astype('int')  # 应用尺度缩放
        
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint['orientation']  # 使用字典访问方式
        cos_angle = math.cos(math.radians(angle))
        sin_angle = math.sin(math.radians(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))
        
        # 计算描述符窗口大小
        hist_width = scale_multiplier * 0.5 * scale * keypoint['size']  # 使用字典访问方式
        half_width = int(round(hist_width * math.sqrt(2) * (window_width + 1) * 0.5))
        half_width = int(min(half_width, math.sqrt(num_rows ** 2 + num_cols ** 2)))
        
        # 在关键点周围区域计算梯度
        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                # 旋转坐标
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                
                # 计算在直方图中的位置
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                
                if -1 < row_bin < window_width and -1 < col_bin < window_width:
                    # 计算实际图像坐标
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    
                    # 确保坐标在图像范围内
                    if 0 < window_row < num_rows - 1 and 0 < window_col < num_cols - 1:
                        # 计算梯度
                        if hasattr(gaussian_image, 'getpixel'):
                            # PIL 图像使用 getpixel
                            dx = gaussian_image.getpixel((window_col + 1, window_row)) - gaussian_image.getpixel((window_col - 1, window_row))
                            dy = gaussian_image.getpixel((window_col, window_row - 1)) - gaussian_image.getpixel((window_col, window_row + 1))
                        else:
                            # NumPy 数组使用索引
                            dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                            dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        
                        gradient_magnitude = math.sqrt(dx * dx + dy * dy)
                        gradient_orientation = math.degrees(math.atan2(dy, dx)) % 360
                        
                        # 计算权重
                        weight = math.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        
                        # 存储计算结果
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)
        
        # 构建方向直方图
        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            # 三线性插值
            row_bin_floor = math.floor(row_bin)
            col_bin_floor = math.floor(col_bin)
            orientation_bin_floor = math.floor(orientation_bin)
            
            row_fraction = row_bin - row_bin_floor
            col_fraction = col_bin - col_bin_floor
            orientation_fraction = orientation_bin - orientation_bin_floor
            
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins
            
            # 计算插值权重
            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)
            
            # 更新直方图
            orientation_bin_floor = int(orientation_bin_floor)
            next_orientation_bin = (orientation_bin_floor + 1) % num_bins
            
            histogram_tensor[int(row_bin_floor) + 1, int(col_bin_floor) + 1, orientation_bin_floor] += c000
            histogram_tensor[int(row_bin_floor) + 1, int(col_bin_floor) + 1, next_orientation_bin] += c001
            histogram_tensor[int(row_bin_floor) + 1, int(col_bin_floor) + 2, orientation_bin_floor] += c010
            histogram_tensor[int(row_bin_floor) + 1, int(col_bin_floor) + 2, next_orientation_bin] += c011
            histogram_tensor[int(row_bin_floor) + 2, int(col_bin_floor) + 1, orientation_bin_floor] += c100
            histogram_tensor[int(row_bin_floor) + 2, int(col_bin_floor) + 1, next_orientation_bin] += c101
            histogram_tensor[int(row_bin_floor) + 2, int(col_bin_floor) + 2, orientation_bin_floor] += c110
            histogram_tensor[int(row_bin_floor) + 2, int(col_bin_floor) + 2, next_orientation_bin] += c111
        
        # 生成描述符向量
        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()
        
        # 归一化处理
        threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        norm_value = max(np.linalg.norm(descriptor_vector), float_tolerance)
        descriptor_vector /= norm_value
        
        # 转换为0-255范围
        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector = np.clip(descriptor_vector, 0, 255)
        
        descriptors.append(descriptor_vector)
    
    return np.array(descriptors, dtype='float32')











