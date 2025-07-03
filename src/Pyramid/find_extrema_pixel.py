import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import math

def is_pixel_extremum(first_sub, second_sub, third_sub, threshold):
    """
    判断一个像素是否是尺度空间中的极值点（极大值或极小值）
    
    参数:
    first_sub (np.ndarray): 上一尺度的3x3邻域
    second_sub (np.ndarray): 当前尺度的3x3邻域
    third_sub (np.ndarray): 下一尺度的3x3邻域
    threshold (float): 对比度阈值
    
    返回:
    bool: 如果是极值点返回True，否则返回False
    
    判断逻辑:
    1. 检查中心像素值是否超过对比度阈值
    2. 如果是正极值，检查是否大于所有26个邻域点
    3. 如果是负极值，检查是否小于所有26个邻域点
    
    在SIFT算法中的作用:
    这是关键点检测的第一步，用于初步筛选可能的特征点位置
    """
    center_val = second_sub[1, 1]
    
    # 检查是否超过对比度阈值（过滤弱响应点）
    if abs(center_val) <= threshold:
        return False
    
    # 判断极大值
    if center_val > 0:
        # 检查所有邻域点（包括三个尺度）
        return (center_val >= first_sub).all() and \
               (center_val >= third_sub).all() and \
               (center_val >= second_sub[0, :]).all() and \
               (center_val >= second_sub[2, :]).all() and \
               center_val >= second_sub[1, 0] and \
               center_val >= second_sub[1, 2]
    
    # 判断极小值
    elif center_val < 0:
        return (center_val <= first_sub).all() and \
               (center_val <= third_sub).all() and \
               (center_val <= second_sub[0, :]).all() and \
               (center_val <= second_sub[2, :]).all() and \
               center_val <= second_sub[1, 0] and \
               center_val <= second_sub[1, 2]
    
    return False

def compute_gradient_at_center_pixel(pixel_cube):
    """
    计算3x3x3像素块中心点的梯度向量 (dx, dy, ds)
    使用中心差分法（二阶精度）：
    $$\frac{\partial f}{\partial x} \approx \frac{f(x+h,y,s) - f(x-h,y,s)}{2h}$$
    其中 $h=1$（像素间距），其他方向同理。
    
    参数:
    pixel_cube (ndarray): 3x3x3的像素值数组
    
    返回:
    ndarray: 梯度向量 [dx, dy, ds]
    
    数学原理:
    使用中心差分法（二阶精度）：
        dx = 0.5 * (f(x+1,y,s) - f(x-1,y,s))
        dy = 0.5 * (f(x,y+1,s) - f(x,y-1,s))
        ds = 0.5 * (f(x,y,s+1) - f(x,y,s-1))
    梯度表示像素值在三个维度的变化率，是定位极值点的基础。
    """
    dx = 0.5 * (pixel_cube[1, 1, 2] - pixel_cube[1, 1, 0])
    dy = 0.5 * (pixel_cube[1, 2, 1] - pixel_cube[1, 0, 1])
    ds = 0.5 * (pixel_cube[2, 1, 1] - pixel_cube[0, 1, 1])
    return np.array([dx, dy, ds])

def compute_hessian_at_center_pixel(pixel_cube):
    """
    计算3x3x3像素块中心点的Hessian矩阵
    
    参数:
    pixel_cube (ndarray): 3x3x3的像素值数组
    
    返回:
    ndarray: 3x3 Hessian矩阵
    
    数学基础：
    二阶纯导数使用中心差分：
    $$\frac{\partial^2 f}{\partial x^2} \approx \frac{f(x+h) - 2f(x) + f(x-h)}{h^2}$$
    混合导数使用四角差分：
    $$\frac{\partial^2 f}{\partial x \partial y} \approx \frac{f(x+h,y+h) - f(x+h,y-h) - f(x-h,y+h) + f(x-h,y-h)}{4h^2}$$

    数学原理:
    使用中心差分法计算二阶导数：
        dxx = f(x+1,y,s) - 2f(x,y,s) + f(x-1,y,s)
        dyy = f(x,y+1,s) - 2f(x,y,s) + f(x,y-1,s)
        dss = f(x,y,s+1) - 2f(x,y,s) + f(x,y,s-1)
    混合导数使用四角差分：
        dxy = 0.25 * (f(x+1,y+1,s) - f(x+1,y-1,s) - f(x-1,y+1,s) + f(x-1,y-1,s))
    Hessian矩阵描述函数曲率，用于精确定位极值点位置
    """
    center_value = pixel_cube[1, 1, 1]
    dxx = pixel_cube[1, 1, 2] - 2 * center_value + pixel_cube[1, 1, 0]
    dyy = pixel_cube[1, 2, 1] - 2 * center_value + pixel_cube[1, 0, 1]
    dss = pixel_cube[2, 1, 1] - 2 * center_value + pixel_cube[0, 1, 1]
    dxy = 0.25 * (pixel_cube[1, 2, 2] - pixel_cube[1, 2, 0] - pixel_cube[1, 0, 2] + pixel_cube[1, 0, 0])
    dxs = 0.25 * (pixel_cube[2, 1, 2] - pixel_cube[2, 1, 0] - pixel_cube[0, 1, 2] + pixel_cube[0, 1, 0])
    dys = 0.25 * (pixel_cube[2, 2, 1] - pixel_cube[2, 0, 1] - pixel_cube[0, 2, 1] + pixel_cube[0, 0, 1])
    return np.array([
        [dxx, dxy, dxs],
        [dxy, dyy, dys],
        [dxs, dys, dss]
    ])

def localize_extremum_via_quadratic_fit(i, j, layer_idx, octave_idx, num_intervals, dog_octave, sigma, contrast_threshold, border_width, eigenvalue_ratio=10, max_attempts=5):
    """
    通过二次拟合精确定位极值点位置（亚像素级）
    
    参数:
    i, j (int): 初始整数坐标
    layer_idx (int): 当前层索引（中间层）
    octave_idx (int): 当前组索引
    num_intervals (int): 每组间隔数
    dog_octave (list): 当前组的DoG图像列表
    sigma (float): 基础sigma
    contrast_threshold (float): 对比度阈值
    border_width (int): 图像边界宽度
    eigenvalue_ratio (float): 曲率阈值比例（默认10）
    max_attempts (int): 最大迭代次数（默认5）
    
    返回:
    tuple: (精确定位后的关键点, 最终层索引) 或 None（定位失败时）
    
    数学原理:
    1. 在(i,j,layer_idx)位置构建二次函数模型:
       f(x) ≈ f(0) + ∇f·x + 1/2 xᵀHx
    2. 极值点位置: x* = -H⁻¹∇f
    3. 迭代更新位置直到收敛
    """
    # 初始化变量
    extremum_outside = False
    img_shape = dog_octave[0].shape
    
    for attempt in range(max_attempts):
        # 1. 构建3x3x3像素块（转换为float32并归一化）
        first_img = dog_octave[layer_idx-1]
        second_img = dog_octave[layer_idx]
        third_img = dog_octave[layer_idx+1]
        
        # 提取3x3区域（注意边界检查）
        first_patch = first_img[i-1:i+2, j-1:j+2].astype(np.float32) / 255.0
        second_patch = second_img[i-1:i+2, j-1:j+2].astype(np.float32) / 255.0
        third_patch = third_img[i-1:i+2, j-1:j+2].astype(np.float32) / 255.0
        
        # 构建3x3x3数组
        pixel_cube = np.stack([first_patch, second_patch, third_patch])
        
        # 2. 计算梯度和Hessian
        gradient = compute_gradient_at_center_pixel(pixel_cube)
        hessian = compute_hessian_at_center_pixel(pixel_cube)
        
        # 3. 求解更新向量：δ = -H⁻¹∇
        update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        
        # 4. 检查收敛（更新量小于0.5像素）
        # 如果所有方向上的偏移量都小于0.5像素，则认为已达到收敛
        if np.max(np.abs(update)) < 0.5:
            break
            
        # 5. 更新位置
        j += int(round(update[0]))
        i += int(round(update[1]))
        layer_idx += int(round(update[2]))
        
        # 6. 边界检查
        # 确保更新后的点仍在图像边界内，如果超出边界，则放弃该点
        if (i < border_width or i >= img_shape[0] - border_width or 
            j < border_width or j >= img_shape[1] - border_width or
            layer_idx < 1 or layer_idx > num_intervals):
            extremum_outside = True
            break
            
    # 检查迭代失败情况
    if extremum_outside or attempt == max_attempts - 1:
        # print('Updated extremum moved outside of image before reaching convergence. Skipping...')
        # print('Exceeded maximum number of attempts without reaching convergence for this extremum. Skipping...')
        return None
        
    # 7. 计算更新后的函数值
    # 这个对应的是响应值response，响应值高，表示该关键点与周围区域差异很大（很"突出"）。
    updated_value = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, update)
    
    # 8. 对比度检查
    if abs(updated_value) * num_intervals < contrast_threshold:
        return None
        
    # 9. 曲率检查（边缘抑制）
    xy_hessian = hessian[:2, :2]
    trace_hessian = np.trace(xy_hessian)
    det_hessian = np.linalg.det(xy_hessian)
    
    if det_hessian <= 0 or eigenvalue_ratio * (trace_hessian ** 2) >= ((eigenvalue_ratio + 1) ** 2) * det_hessian:
        return None
        
    # 10. 构建关键点
    keypoint = {
        'class_id': -1,
        'x': (j + update[0]) * (2 ** octave_idx),  # 还原到原始图像坐标
        'y': (i + update[1]) * (2 ** octave_idx),
        'octave': octave_idx+layer_idx*(2**8)+int(round((update[2]+0.5)*255))*(2**16),
        'layer': layer_idx,
        'size': sigma * (2 ** ((layer_idx + update[2]) / float(num_intervals))) * (2 ** (octave_idx + 1)), # octave_index + 1 because the input image was doubled
        'response': abs(updated_value)
    }
    return keypoint, layer_idx

def compute_keypoints_with_orientations(keypoint, octave_idx, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    """
    计算每个关键点的方向
    
    参数:
    keypoint (dict): 关键点信息字典
    octave_idx (int): 关键点所在的组索引
    gaussian_image (PIL.Image or numpy.ndarray): 关键点所在的高斯模糊图像
    radius_factor (float): 用于确定梯度计算区域半径的因子
    num_bins (int): 方向直方图的柱数
    peak_ratio (float): 次峰值相对于主峰值的比例阈值
    scale_factor (float): 尺度因子，间接控制了用于计算方向直方图时的邻域大小
    
    返回:
    list: 带有方向信息的关键点列表
    
    处理流程:
    1. 计算关键点周围区域的梯度方向和幅值
    2. 使用高斯加权累积到方向直方图中
    3. 平滑直方图
    4. 找到直方图中的峰值
    5. 对每个峰值创建一个新的关键点
    
    在SIFT算法中的作用:
    为每个关键点分配一个或多个主方向，使特征具有旋转不变性
    """
    keypoints_with_orientations = []
    # 确保gaussian_image是numpy数组
    if not isinstance(gaussian_image, np.ndarray):
        gaussian_image = np.array(gaussian_image)
    image_shape = gaussian_image.shape
    
    # 计算尺度
    scale = scale_factor * keypoint['size'] / float(2 ** (octave_idx + 1))
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2) # 权重因子，用于高斯加权
    
    # 初始化直方图
    raw_histogram = np.zeros(num_bins)
    smooth_histogram = np.zeros(num_bins)
    
    # 获取关键点在当前组中的坐标
    # keypoint['x']和keypoint['y']是相对于原始图像(第0组)的坐标
    # 因此需要将它们转换为当前组的坐标，第几组就除以2的几次方
    region_center_x = int(round(keypoint['x'] / float(2 ** octave_idx)))
    region_center_y = int(round(keypoint['y'] / float(2 ** octave_idx)))
    
    # 计算区域内每个像素的梯度方向和幅值
    for i in range(-radius, radius + 1):
        region_y = region_center_y + i
        if 0 < region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = region_center_x + j
                if 0 < region_x < image_shape[1] - 1:
                    # 计算梯度
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    
                    # 计算梯度幅值（梯度强度决定投票数量）和方向（角度）
                    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                    
                    # 高斯加权
                    # 距离越远，权重越小（指数衰减）
                    # 公式exp(k * r²)是高斯函数的简化形式
                    # 中心点权重最大（r=0时weight=1），随距离增加权重指数衰减
                    distance_from_center = np.sqrt(i ** 2 + j ** 2)
                    weight = np.exp(weight_factor * (distance_from_center ** 2))
                    
                    # 累积到直方图中
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    histogram_index = histogram_index % num_bins
                    raw_histogram[histogram_index] += weight * gradient_magnitude
    
    # 平滑直方图
    # (6*自己 + 4*左邻居 + 4*右邻居 + 1*左左邻居 + 1*右右邻居)/16
    # 防止某个候选人因偶然因素高票
    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 
                               4 * raw_histogram[n - 1] + 
                               4 * raw_histogram[(n + 1) % num_bins] + 
                               1 * raw_histogram[n - 2] + 
                               1 * raw_histogram[(n + 2) % num_bins]
                              ) / 16.
    
    # 找到直方图中的峰值
    orientation_max = np.max(smooth_histogram)
    
    # 找到所有峰值（比左右两侧都大的点）
    # 找出所有满足条件（比左右两侧都大的点）的索引位置
    orientation_peaks_index = np.where(
        np.logical_and(
            smooth_histogram > np.roll(smooth_histogram, 1),
            smooth_histogram > np.roll(smooth_histogram, -1)
        )
    )[0]
    
    # 处理每个峰值
    for peak_index in orientation_peaks_index:
        peak_value = smooth_histogram[peak_index]
        
        # 只处理大于主峰值一定比例的峰值
        if peak_value >= peak_ratio * orientation_max:
            # 二次插值确定峰值的精确位置
            # 使用% num_bins=36处理循环边界，当峰值在索引0（0°）时，左侧邻居是索引35（350°）
            left_value = smooth_histogram[(peak_index - 1) % num_bins] # 峰值左侧柱的值
            right_value = smooth_histogram[(peak_index + 1) % num_bins] # 峰值右侧柱的值
            
            # 二次插值公式
            # 像跷跷板一样，中间的支点承重，两端体重，要让支点承重使得平衡，那这个平衡位置就是最终的峰值。
            # 移动量 = (左边体重 - 右边体重) / (左边体重 + 右边体重 - 2×支点承重)
            # 0.5系数，是归一化因子，限制偏移量在[-0.5, 0.5]范围内，因为是对index来进行偏移的。
            interpolated_peak_index = (peak_index + 
                                      (0.5 * (left_value - right_value)) / 
                                      (left_value - 2 * peak_value + right_value)) % num_bins
            
            # 计算方向（角度）
            # 360减去，数学计算中角度使用顺时针方向表示，但图像坐标系中Y轴是向下的（与数学坐标系相反）
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < 1e-7:  # 浮点数容差
                orientation = 0
            
            # 创建带方向的新关键点
            new_keypoint = keypoint.copy()
            new_keypoint['orientation'] = orientation
            keypoints_with_orientations.append(new_keypoint)
    
    return keypoints_with_orientations

def find_scale_space_extrema(gaussian_pyramid, dog_pyramid, sigma, num_intervals, border_width, contrast_thresh=0.04, candidate_num_keypoints=0, final_num_keypoints=0):
    """
    在DoG金字塔中检测尺度空间极值点（关键点）
    
    参数:
    gaussian_pyramid (list): 高斯金字塔 [组][层]
    dog_pyramid (list): DoG金字塔 [组][层]
    gaussian_sigmas (list): 高斯金字塔每层所需的模糊核sigma值
    num_intervals (int): 每组中的间隔数
    border_width (int): 图像边界宽度（避免在边界检测）
    contrast_thresh (float): 对比度阈值（默认0.04）
    candidate_num_keypoints (int): 最大候选关键点数限制（默认0，不限制）
    final_num_keypoints (int): 最终关键点数限制（默认0，不限制）

    返回:
    list: 检测到的关键点列表
    
    处理流程:
    1. 计算阈值：threshold = floor(0.5 * contrast_thresh / num_intervals * 255)
    2. 遍历每个组(octave)和每层DoG图像
    3. 在每层图像中，遍历每个像素（跳过边界）
    4. 检查当前像素是否是3x3x3邻域中的极值点
    5. 如果是，尝试精确定位关键点位置（通过二次拟合）
    6. 计算关键点方向
    7. 返回所有关键点
    
    在SIFT算法中的作用:
    这是SIFT特征检测的核心步骤，用于定位稳定的特征点
    """    
    # 计算阈值（来自OpenCV实现）
    threshold = math.floor(0.5 * contrast_thresh / num_intervals * 255)
    # 最终关键点列表
    keypoints = []
    
    # 遍历每个组
    for octave_idx, dog_octave in enumerate(dog_pyramid):
        # 每组有 len(dog_octave) 层，我们取连续的三层进行检测
        for layer_idx in range(len(dog_octave) - 2):
            # 中间层索引
            middle_layer_idx = layer_idx + 1

            # 获取三个连续的DoG层
            first = dog_octave[middle_layer_idx-1]
            second = dog_octave[middle_layer_idx]
            third = dog_octave[middle_layer_idx + 1]
            
            # 获取图像尺寸
            height, width = second.shape
            
            # 遍历图像中的每个像素（跳过边界）
            for i in range(border_width, height - border_width):
                for j in range(border_width, width - border_width):
                    # 提取3x3x3邻域
                    first_region = first[i-1:i+2, j-1:j+2]
                    second_region = second[i-1:i+2, j-1:j+2]
                    third_region = third[i-1:i+2, j-1:j+2]
                    
                    # 检查是否是极值点
                    if is_pixel_extremum(first_region, second_region, third_region, threshold):
                        # 精确定位关键点
                        # localization_result = localize_extremum_via_quadratic_fit(
                        #     i, j, middle_layer_idx, octave_idx, num_intervals, dog_octave,
                        #     gaussian_sigmas[middle_layer_idx], contrast_thresh, border_width
                        # )
                        localization_result = localize_extremum_via_quadratic_fit(
                            i, j, middle_layer_idx, octave_idx, num_intervals, dog_octave,
                            sigma, contrast_thresh, border_width
                        )
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            # keypoints.append(keypoint)
                            keypoints_with_orientations = compute_keypoints_with_orientations(keypoint, octave_idx, gaussian_pyramid[octave_idx][localized_image_index])
                            for keypoint_with_orientation in keypoints_with_orientations:                              
                                keypoints.append(keypoint_with_orientation)
    #                             # 检查是否达到最大候选关键点数限制
    #                             if len(keypoints) == candidate_num_keypoints:
    #                                 # 按响应值降序排序
    #                                 keypoints.sort(key=lambda x: x['response'], reverse=True)     
    #                                 # 应用关键点数量限制
    #                                 if final_num_keypoints > 0:
    #                                     return keypoints[:final_num_keypoints]
    #                                 else:
    #                                     return keypoints
    # # 如果未达到最大候选关键点数限制，则按响应值降序排序
    # keypoints.sort(key=lambda x: x['response'], reverse=True)
    # if final_num_keypoints > 0:
    #     return keypoints[:final_num_keypoints]
    return keypoints

def visualize_keypoints(image, keypoints, title="检测到的SIFT关键点"):
    """
    在图像上可视化关键点
    
    参数:
    image (PIL.Image): 原始图像
    keypoints (list): 关键点列表，每个关键点是一个字典
    title (str): 图像标题
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
        # 注意：keypoint坐标是相对于原始图像的，但可能需要调整
        # 因为SIFT处理过程中图像可能被放大了2倍
        x, y = kp['x'], kp['y']
        
        # 确保坐标在图像范围内
        if 0 <= x < width and 0 <= y < height:
            size = kp['size'] / 2  # 圆的半径
            
            # 绘制圆圈表示关键点位置和大小
            circle = plt.Circle((x, y), size, color='r', fill=False, linewidth=1.5)
            plt.gca().add_patch(circle)
            
            # 可选：绘制关键点中心
            plt.plot(x, y, 'r.', markersize=3)
    
    plt.title(f"{title} - 共{len(keypoints)}个关键点")
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=True)