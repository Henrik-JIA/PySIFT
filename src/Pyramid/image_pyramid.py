import numpy as np
from PIL import Image, ImageFilter
from scipy.signal import convolve2d
from scipy import ndimage
import matplotlib.pyplot as plt
import math
import logging
import cv2
logger = logging.getLogger(__name__)
float_tolerance = 1e-7

def bilinear_interpolation(image, scale_factor):
    """
    实现与OpenCV cv2.resize(INTER_LINEAR)相同效果的双线性插值
    
    参数:
    image (np.array): 输入图像 (浮点型)
    scale_factor (float): 缩放因子
    
    返回:
    np.array: 插值后的图像 (浮点型)
    """
    # 确保输入图像是浮点型
    image = image.astype(np.float32)
    
    # 获取图像的尺寸
    height, width = image.shape
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    new_img = np.zeros((new_height, new_width), dtype=np.float32)
    
    inv_scale = 1.0 / scale_factor

    for i in range(new_height):
        for j in range(new_width):
            # OpenCV式的坐标映射：考虑像素中心点
            # 公式: src_x = (dst_x + 0.5) * inv_scale - 0.5
            src_x = (i + 0.5) * inv_scale - 0.5
            src_y = (j + 0.5) * inv_scale - 0.5
            
            # 计算四个最近邻点的坐标
            x0 = int(np.floor(src_x))
            x1 = min(x0 + 1, height - 1)
            y0 = int(np.floor(src_y))
            y1 = min(y0 + 1, width - 1)
            
            # 处理边界情况
            x0 = max(0, x0)
            y0 = max(0, y0)
            
            # 计算插值权重
            wx = src_x - x0
            wy = src_y - y0
            
            # 双线性插值计算
            value = (1 - wx) * (1 - wy) * image[x0, y0] + \
                    wx * (1 - wy) * image[x1, y0] + \
                    (1 - wx) * wy * image[x0, y1] + \
                    wx * wy * image[x1, y1]
            
            new_img[i, j] = value

    return new_img

def create_base_image(image, target_sigma, initial_blur):
    """
    创建基础图像：上采样2倍并应用精确的高斯模糊
    
    参数:
    image (PIL.Image): 输入图像
    target_sigma (float): 目标模糊级别
    initial_blur (float): 原始图像已存在的模糊量
    
    返回:
    PIL.Image: 处理后的基础图像

    解释：
    原始图像自带模糊 σ=0.5
        [512x512] 
        │
        ▼ 放大2倍
    中间图像 (upsampled)
        [1024x1024] → 继承模糊等效为 2×0.5=1.0 (因像素密度减半)
        │
        ▼ 添加模糊 σ_diff=1.25
    总模糊图像 (base_image) = √(1.0² + 1.25²) = 1.6
        [1024x1024] → 达到目标模糊度σ=1.6

    直接对原始图像应用 σ_total = √[(2×σ_initial)² + σ_diff²] = √[1.0² + 1.25²] = 1.6 模糊效果在数学等效性上是成立的。
    但：图像分辨率是不同的
    原始图像 (512x512@σ=0.5)
    │
    ├─ 路径A：直接模糊σ=1.6 → 得到512x512@σ=1.6 (丢失小尺度特征)
    │
    └─ 路径B：SIFT标准流程
        1. 放大2倍 → 1024x1024@等效σ=1.0
        2. 追加模糊σ_diff=1.25 → 1024x1024@σ=1.6 
        (保留小尺度特征能力)
    """    
    # 将PIL图像转换为NumPy数组
    image = np.array(image).astype('float32')

    # 1. 上采样2倍（双线性插值）
    # upsampled = bilinear_interpolation(image, 2) # 这个自己写的插值函数也可以实现同样的结果
    upsampled = ndimage.zoom(image, (2, 2), order=1, grid_mode=True, mode='grid-mirror')
    
    # 2. 计算需要添加的模糊量
    # 公式: σ_target² = (2×σ_initial)²) + σ_diff²
    # σ_diff = √(σ_target² - (2×σ_initial)²)
    diff = target_sigma**2 - (2 * initial_blur)**2
    required_blur = math.sqrt(max(diff, 0.01))
    
    # 3. 应用高斯模糊
    base_image = ndimage.gaussian_filter(upsampled, sigma=required_blur, truncate=4, mode='mirror')
    # base_image = cv2.GaussianBlur(upsampled, (0, 0), sigmaX=required_blur, sigmaY=required_blur) 
    
    return base_image

def compute_number_of_octaves(image):
    """
    计算图像金字塔的组数(octaves)
    
    参数:
    image (np.array): 输入图像
    
    返回:
    int: 金字塔的组数
    
    公式:
    octaves = log₂(min_dimension) - 1
        
    解释:
    1. 金字塔每组图像尺寸减半，min_dimension：取图像短边长度（如 512x640 的图像取 512）
    2. log₂(512)：计算 512 用 2 的幂次表示（2⁹=512 → 得 9）
    3. 减 1：预留安全余量（避免图像过小导致特征失效）
    4. 最小尺寸限制：当图像尺寸小于约 8-12 像素时停止
    5. 公式确保金字塔有足够层数，同时避免过小的图像
    6. 取整：金字塔组数必须是整数
    
    示例:
    对于 512x512 图像:
    min_dimension = 512
    log₂(512) = 9
    octaves = 9 - 1 = 8
    """
    # 获取图像的尺寸
    image_shape = image.shape

    # 获取图像的最小维度
    min_dimension = min(image_shape)
    
    # 计算以2为底的对数
    log_value = math.log(min_dimension, 2)
    
    # 应用公式并取整
    octaves = int(round(log_value - 1))
    
    return max(octaves, 1)  # 确保至少1组 

def generate_gaussian_kernel_sigmas(sigma, num_intervals):
    """
    生成高斯金字塔每层所需的模糊核sigma值
    
    参数:
    sigma (float): 基础层的sigma值（通常为1.6）
    num_intervals (int): 每组(octave)中的层数间隔（通常为3）
    
    返回:
    numpy.ndarray: 高斯核sigma值数组，长度为 num_intervals + 3
    
    公式推导:
    1. 每组的层数 = num_intervals + 3
    2. 尺度因子 k = 2^(1/num_intervals)
    3. 第i层的总模糊量: σ_i = σ * k^i
    4. 相邻层间需要添加的模糊量: 
        σ_diff = √(σ_i² - σ_{i-1}²)
    
    在SIFT算法中的作用:
    构建高斯金字塔时，这些值用于计算每层需要添加的高斯模糊量
    量
    kernel_sigma 已经是增量模糊量（Δσ），不是总模糊量：
    第0层：总模糊 = σ₀ (已包含在base_image中)
    第1层：只需添加 Δσ₁ 即可达到总模糊 √(σ₀² + Δσ₁²) = k·σ₀
    第2层：在已有k·σ₀基础上添加 Δσ₂ 达到 √((kσ₀)² + Δσ₂²) = k²·σ₀
    """
    # 计算每组的图像数量
    num_images_per_octave = num_intervals + 3
    
    # 计算尺度因子 k
    k = 2 ** (1.0 / num_intervals)
    
    # 初始化高斯核数组
    gaussian_kernels = np.zeros(num_images_per_octave)
    
    # 第一层使用基础sigma值
    gaussian_kernels[0] = sigma
    
    # 计算后续层的模糊量
    for image_index in range(1, num_images_per_octave):
        # 前一层的总模糊量
        sigma_previous = (k ** (image_index - 1)) * sigma
        
        # 当前层的目标总模糊量
        sigma_total = k * sigma_previous
        
        # 需要添加的模糊量
        gaussian_kernels[image_index] = math.sqrt(
            sigma_total ** 2 - sigma_previous ** 2
        )
    
    return gaussian_kernels

def build_gaussian_pyramid(base_image, num_octaves, gaussian_kernel_sigmas):
    """
    构建高斯金字塔
    
    参数:
    base_image (np.array): 基础图像（已上采样2倍和高斯模糊）
    num_octaves (int): 金字塔的组数
    gaussian_kernels (np.array): 高斯核sigma值数组
    
    返回:
    list: 高斯金字塔，结构为 [组][层]
    
    金字塔结构:
    1. 每组包含 len(gaussian_kernels) 层
    2. 每组的第一层是上一组的倒数第三层降采样得到
    3. 每组内部通过连续模糊构建
    """
    # 初始化金字塔
    gaussian_pyramid = []
    
    # 当前处理的图像（初始为基础图像）
    current_image = base_image
    
    # 遍历所有组
    for octave_index in range(num_octaves):
        # 初始化当前组
        octave_images = []
        
        # 添加第一层（已有基础模糊）
        octave_images.append(current_image)
        
        # 应用高斯核生成当前组的其他层
        for kernel_sigma in gaussian_kernel_sigmas[1:]:
            # 应用高斯模糊，使用ndimage.gaussian_filter
            current_image = ndimage.gaussian_filter(
                current_image, 
                sigma=kernel_sigma, 
                truncate=4.0,  # 默认值，可以根据需要调整
                mode='mirror'  # 类似于OpenCV的BORDER_REFLECT_101
            )
            # current_image = cv2.GaussianBlur(current_image, (0, 0), sigmaX=kernel_sigma, sigmaY=kernel_sigma) 

            octave_images.append(current_image)
        
        # 将当前组添加到金字塔
        gaussian_pyramid.append(octave_images)
        
        # 准备下一组的基础图像
        # 取当前组的倒数第三层（索引为-3）进行降采样
        if octave_index < num_octaves - 1:  # 最后一组不需要降采样
            base_for_next = octave_images[-3]
            # 降采样2倍，使用ndimage.zoom替代PIL的resize
            # 注意：zoom的scale_factor是目标尺寸/原始尺寸，所以这里是0.5
            current_image = ndimage.zoom(
                base_for_next,
                (0.5, 0.5),  # 降采样2倍，所以是0.5
                order=1,      # order=0对应最近邻插值，1表示双线性插值，2表示双三次插值
                grid_mode=True,  # 考虑像素中心
                mode='grid-mirror'  # 类似于OpenCV的边界处理
            )
            # current_image = cv2.resize(base_for_next, (int(base_for_next.shape[1] / 2), int(base_for_next.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)

    return gaussian_pyramid


def build_dog_pyramid(gaussian_pyramid):
    """
    构建高斯差分金字塔(DoG)，用于近似拉普拉斯算子
    
    参数:
    gaussian_pyramid (list): 高斯金字塔，结构为 [组][层]
    
    返回:
    list: DoG金字塔，结构为 [组][层]，每组的层数比高斯金字塔少1
    
    计算原理:
    DoG = 高斯金字塔中相邻层的差值
    DoG(x,y,σ) = L(x,y,kσ) - L(x,y,σ)
    
    在SIFT算法中的作用:
    1. 近似拉普拉斯算子，用于检测关键点
    2. 比直接计算拉普拉斯更高效
    3. 对尺度变化具有不变性
    """
    logger.debug('Generating Difference-of-Gaussian images...')
    dog_pyramid = []
    
    # # ====================================
    # # 遍历每组高斯金字塔
    # for octave_index, gaussian_octave in enumerate(gaussian_pyramid):
    #     dog_octave = []
        
    #     # 计算当前组的DoG图像
    #     for layer_index in range(len(gaussian_octave) - 1):
    #         # 将PIL图像转换为NumPy数组
    #         img1 = gaussian_octave[layer_index]
    #         img2 = gaussian_octave[layer_index + 1]
            
    #         # 计算差分（避免无符号整数环绕问题）
    #         dog_image = img2 - img1
            
    #         # 添加到当前组
    #         dog_octave.append(dog_image)
        
    #     # 将当前组添加到金字塔
    #     dog_pyramid.append(dog_octave)
    
    # return dog_pyramid

    # ====================================
    # 遍历每组高斯金字塔，与上面注释的代码等效
    for gaussian_octave in gaussian_pyramid:
        # 使用向量化操作一次性计算整个组的DoG图像
        dog_octave = [gaussian_octave[i+1] - gaussian_octave[i] for i in range(len(gaussian_octave)-1)]
        
        # 将当前组添加到金字塔
        dog_pyramid.append(dog_octave)
    
    return dog_pyramid
    # ====================================

def visualize_pyramids(gaussian_pyramid, dog_pyramid, octave_indices=None):
    """
    可视化金字塔结构
    
    参数:
    gaussian_pyramid (list): 高斯金字塔
    dog_pyramid (list): DoG金字塔
    octave_indices (list, optional): 要显示的组索引列表，None表示显示所有组
    """  
    # 确定要显示的组
    if octave_indices is None:
        octave_indices = range(len(gaussian_pyramid))
    else:
        # 确保索引在有效范围内
        octave_indices = [idx for idx in octave_indices 
                          if 0 <= idx < len(gaussian_pyramid)]
    
    # 可视化高斯金字塔
    plt.figure(figsize=(13, 10))
    plt.suptitle('Gaussian Pyramid')
    plot_index = 1
    for octave_idx in octave_indices:
        octave = gaussian_pyramid[octave_idx]
        for layer_idx, img in enumerate(octave):
            plt.subplot(len(octave_indices), len(octave), plot_index)
            plt.imshow(np.array(img), cmap='gray')
            plt.title(f'Octave_idx:{octave_idx} \n Gaussian_Layer_idx:{layer_idx}', fontsize=10)
            plt.axis('off')
            plot_index += 1
    plt.tight_layout()
    
    # 可视化DoG金字塔
    plt.figure(figsize=(13, 10))
    plt.suptitle('Difference of Gaussian Pyramid')
    plot_index = 1
    for octave_idx in octave_indices:
        octave = dog_pyramid[octave_idx]
        for layer_idx, img in enumerate(octave):
            plt.subplot(len(octave_indices), len(octave), plot_index)
            plt.imshow(img, cmap='gray')
            # 明确指出是哪两层高斯图像相减
            # DoG[i] = Gaussian[i+1] - Gaussian[i]
            title = f'Octave_idx:{octave_idx},DoG_Layer_idx:{layer_idx}'
            subtitle = f'Gaussian[{layer_idx+1}] - Gaussian[{layer_idx}]'
            plt.title(f'{title}\n{subtitle}', fontsize=10)
            plt.axis('off')
            plot_index += 1
    plt.tight_layout()
    
    plt.show(block=True)





    