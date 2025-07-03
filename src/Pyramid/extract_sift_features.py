from src.Pyramid.image_pyramid import create_base_image, compute_number_of_octaves, generate_gaussian_kernel_sigmas, build_gaussian_pyramid, build_dog_pyramid, visualize_pyramids
from src.Pyramid.find_extrema_pixel import find_scale_space_extrema, visualize_keypoints
from src.Pyramid.clean_up_keypoints import remove_duplicate_keypoints, convert_keypoints_to_input_image_size, visualize_converted_keypoints
from src.Pyramid.generate_descriptors import generate_descriptors

def extract_sift_features(image):
    """
    从输入图像中提取SIFT特征（关键点和描述符）
    
    参数:
        image (PIL.Image): 输入图像
    
    返回:
        keypoints (list): 关键点列表
        descriptors (numpy.ndarray): 描述符数组
    """
    # ===== 构建图像金字塔 =====
    # 1. 创建基础图像
    base_image = create_base_image(image=image, target_sigma=1.6, initial_blur=0.5)

    # 2. 计算金字塔组数
    num_octaves = compute_number_of_octaves(base_image.size)
    print(f"金字塔组数: {num_octaves}")

    # 3. 生成高斯核
    gaussian_sigmas = generate_gaussian_kernel_sigmas(sigma=1.6, num_intervals=3)

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
    keypoints = find_scale_space_extrema(gaussian_pyramid, dog_pyramid, sigma=1.6, num_intervals=3, border_width=5, candidate_num_keypoints=3000, final_num_keypoints=1000)    
    
    # 8. 可视化关键点
    # visualize_keypoints(image, keypoints, f"图片 #{idx+1} 的SIFT关键点")  

    # 9. 清理关键点
    keypoints = remove_duplicate_keypoints(keypoints)   

    # 10. 转换关键点到输入图像尺寸
    keypoints = convert_keypoints_to_input_image_size(keypoints)    

    # 11. 可视化清理后的关键点
    visualize_converted_keypoints(image, keypoints, f"SIFT keypoints")
    # print(keypoints)
    # ======= 生成描述符 =======
    
    # 12. 生成描述符
    descriptors = generate_descriptors(keypoints, gaussian_pyramid)

    return keypoints, descriptors