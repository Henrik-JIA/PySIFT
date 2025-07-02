import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import math

def create_base_image(image, target_sigma, initial_blur):
    """
    创建基础图像：上采样2倍并应用精确的高斯模糊
    
    参数:
    image (PIL.Image or np.ndarray): 输入图像
    target_sigma (float): 目标模糊级别
    initial_blur (float): 原始图像已存在的模糊量
    
    返回:
    PIL.Image: 处理后的基础图像
    """
    # 如果输入是NumPy数组，转换为PIL图像
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # 1. 上采样2倍（双线性插值）
    new_size = (image.width * 2, image.height * 2)
    upsampled = image.resize(new_size, Image.BILINEAR)
    
    # 2. 计算需要添加的模糊量
    # 公式: σ_target² = (2×σ_initial)²) + σ_diff²
    # σ_diff = √(σ_target² - (2×σ_initial)²)
    required_blur = math.sqrt(
        max(target_sigma**2 - (2 * initial_blur)**2, 0.01)
    )
    
    # 3. 应用高斯模糊
    # 使用PIL的高斯模糊滤波器
    base_image = upsampled.filter(ImageFilter.GaussianBlur(required_blur))
    
    return base_image

def compute_number_of_octaves(image_shape):
    """
    计算图像金字塔的组数(octaves)
    
    参数:
    image_shape (tuple): 图像的尺寸 (height, width)
    
    返回:
    int: 金字塔的组数
    
    公式:
    octaves = log₂(min_dimension) - 1
    
    解释:
    1. 金字塔每组图像尺寸减半
    2. 最小尺寸限制：当图像尺寸小于约 8-12 像素时停止
    3. 公式确保金字塔有足够层数，同时避免过小的图像
    
    示例:
    对于 512x512 图像:
    min_dimension = 512
    log₂(512) = 9
    octaves = 9 - 1 = 8
    """
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
    base_image (PIL.Image): 基础图像（已上采样和模糊）
    num_octaves (int): 金字塔的组数
    gaussian_kernels (np.ndarray): 高斯核sigma值数组
    
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
            # 应用高斯模糊
            current_image = current_image.filter(
                ImageFilter.GaussianBlur(kernel_sigma)
            )
            octave_images.append(current_image)
        
        # 将当前组添加到金字塔
        gaussian_pyramid.append(octave_images)
        
        # 准备下一组的基础图像
        # 取当前组的倒数第三层（索引为-3）进行降采样
        if octave_index < num_octaves - 1:  # 最后一组不需要降采样
            base_for_next = octave_images[-3]
            
            # 降采样2倍（使用最近邻插值）
            new_width = base_for_next.width // 2
            new_height = base_for_next.height // 2
            current_image = base_for_next.resize(
                (new_width, new_height), Image.NEAREST
            )
    
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
    dog_pyramid = []
    
    # 遍历每组高斯金字塔
    for octave_index, gaussian_octave in enumerate(gaussian_pyramid):
        dog_octave = []
        
        # 计算当前组的DoG图像
        for layer_index in range(len(gaussian_octave) - 1):
            # 将PIL图像转换为NumPy数组
            img1 = np.array(gaussian_octave[layer_index], dtype=np.float32)
            img2 = np.array(gaussian_octave[layer_index + 1], dtype=np.float32)
            
            # 计算差分（避免无符号整数环绕问题）
            dog_image = img2 - img1
            
            # 添加到当前组
            dog_octave.append(dog_image)
        
        # 将当前组添加到金字塔
        dog_pyramid.append(dog_octave)
    
    return dog_pyramid

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
    plt.figure(figsize=(15, 10))
    plt.suptitle('高斯金字塔')
    plot_index = 1
    for octave_idx in octave_indices:
        octave = gaussian_pyramid[octave_idx]
        for layer_idx, img in enumerate(octave):
            plt.subplot(len(octave_indices), len(octave), plot_index)
            plt.imshow(np.array(img), cmap='gray')
            plt.title(f'O{octave_idx}L{layer_idx}')
            plt.axis('off')
            plot_index += 1
    plt.tight_layout()
    
    # 可视化DoG金字塔
    plt.figure(figsize=(15, 10))
    plt.suptitle('DoG金字塔')
    plot_index = 1
    for octave_idx in octave_indices:
        octave = dog_pyramid[octave_idx]
        for layer_idx, img in enumerate(octave):
            plt.subplot(len(octave_indices), len(octave), plot_index)
            plt.imshow(img, cmap='gray')
            plt.title(f'O{octave_idx}L{layer_idx}')
            plt.axis('off')
            plot_index += 1
    plt.tight_layout()
    
    plt.show(block=True)





    