import logging

from src.SIFT.image_pyramid import create_base_image, compute_number_of_octaves, generate_gaussian_kernel_sigmas, build_gaussian_pyramid, build_dog_pyramid, visualize_pyramids
from src.SIFT.find_extrema_pixel import find_scale_space_extrema, visualize_keypoints
from src.SIFT.clean_up_keypoints import remove_duplicate_keypoints, convert_keypoints_to_input_image_size, visualize_converted_keypoints
from src.SIFT.generate_descriptors import generate_descriptors

logger = logging.getLogger(__name__)

def extract_sift_features(image, sigma, num_intervals, idx):
    """
    从输入图像中提取SIFT特征（关键点和描述符）
    
    参数:
        image (np.array): 输入图像
    
    返回:
        keypoints (list): 关键点列表
        descriptors (numpy.ndarray): 描述符数组
    """
    # ===== 构建图像金字塔 =====
    # 1. 创建基础图像(上采样2倍（双线性插值），并高斯模糊)
    base_image = create_base_image(image=image, target_sigma=sigma, initial_blur=0.5)

    # 2. 计算金字塔组数
    num_octaves = compute_number_of_octaves(base_image)
    logger.info(f"金字塔组数: {num_octaves}")

    # 3. 生成高斯核
    gaussian_sigmas = generate_gaussian_kernel_sigmas(sigma=sigma, num_intervals=num_intervals)

    # 4. 构建高斯金字塔
    gaussian_pyramid = build_gaussian_pyramid(
        base_image=base_image,
        num_octaves=num_octaves,
        gaussian_kernel_sigmas=gaussian_sigmas
    )

    # 5. 构建DoG金字塔
    dog_pyramid = build_dog_pyramid(gaussian_pyramid)   
    
    # 6. ===== 可选：可视化金字塔 =====
    # 这里可以添加金字塔可视化代码
    # visualize_pyramids(gaussian_pyramid, dog_pyramid, [0,1,2])

    # ============ 检测尺度空间极值点 ============
    # 7. 检测尺度空间极值点
    keypoints = find_scale_space_extrema(gaussian_pyramid, dog_pyramid, sigma=sigma, num_intervals=num_intervals, border_width=5, final_num_keypoints=2000)    
    
    # 8. 可视化关键点(visualize keypoints)
    # visualize_keypoints(image, keypoints, f"Image {idx+1} SIFT keypoints")

    # 9. 清理关键点
    keypoints = remove_duplicate_keypoints(keypoints)

    # 10. 转换关键点到输入图像尺寸
    keypoints = convert_keypoints_to_input_image_size(keypoints)    

    # 11. 可视化清理后的关键点(visualize cleaned keypoints)
    # visualize_converted_keypoints(image, keypoints, f"SIFT converted and cleaned keypoints to input image size")
    # print(keypoints)
    # ======= 生成描述符 =======
    
    # 12. 生成描述符
    descriptors = generate_descriptors(keypoints, gaussian_pyramid)

    return keypoints, descriptors