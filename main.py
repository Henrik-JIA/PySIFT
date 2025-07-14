import sys
import os
import logging
import numpy as np
import cv2
from src.util.image_loader import ImageLoader
from src.SIFT.extract_sift_features import extract_sift_features
from src.Matching.match_descriptor import match_descriptors
from src.Matching.ransac_geometric_consistency import ransac_geometric_consistency, decompose_homography, decompose_fundamental_matrix
from PIL import Image  
import matplotlib.pyplot as plt  

# 创建logs目录
os.makedirs('logs', exist_ok=True)

# 配置根日志记录器
logging.basicConfig(
    level=logging.INFO,  # 全局日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('logs/sift.log', encoding='utf-8'),  # 文件日志
        logging.StreamHandler()  # 控制台日志
    ]
)

# 可以单独设置某些模块的日志级别
logging.getLogger('src.Pyramid.find_extrema_pixel').setLevel(logging.ERROR)
# logging.getLogger('src.Pyramid.find_extrema_pixel').setLevel(logging.DEBUG)

def main():
    """
    主函数：加载并处理图片数据
    """
    # 默认使用项目中的data文件夹
    data_path = "data/Box"
    # data_path = "data/Lenna"
    # data_path = "data/Building"
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        print(f"使用自定义路径: {data_path}")
    else:
        print(f"使用默认路径: {data_path}")
    
    try:
        # 创建图片加载器实例，只加载元数据
        loader = ImageLoader(data_path, load_content=False)
        
        # 获取所有图片元数据
        all_image_data_exif = loader.get_image_data()
        image_count = loader.get_image_count()
        
        print(f"\n共加载 {image_count} 张图片的元数据")
        
        # 处理并显示图片元数据
        process_image_data(all_image_data_exif)
        
    except Exception as e:
        print(f"处理图片时出错: {str(e)}")
        sys.exit(1)

def process_image_data(image_data_exif):
    """
    处理并显示图片数据
    
    参数:
    image_data (list): 包含图片数据的字典列表
    """
    MIN_MATCH_COUNT = 10
    keypoints_list = []
    descriptors_list = []
    imgs_list = []
    for idx, img_exif in enumerate(image_data_exif):
        print(f"图片{idx+1}: {img_exif['path']}")
        print(f"尺寸: {img_exif['width']}x{img_exif['height']}")
        print(f"格式: {img_exif['format']}")
        print(f"大小: {img_exif['size_mb']:.2f} MB")
        # 显示EXIF信息
        display_exif_info(img_exif)
        # 显示GPS信息
        display_gps_info(img_exif)

        # 打开图片并转为灰度图
        image = Image.open(img_exif['path']).convert('L')
        
        imgs_list.append(image)

        # 提取SIFT特征
        keypoints, descriptors = extract_sift_features(image, sigma=1.6, num_intervals=3, idx=idx)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    # ========== 计算相邻图像间的匹配点 ==========
    good_matches_list = []
    for i in range(1,len(descriptors_list)):
        good_matches = match_descriptors(descriptors_list[i-1], descriptors_list[i], ratio_threshold=0.7)
        if len(good_matches) > MIN_MATCH_COUNT:
            good_matches_list.append(good_matches)
        else:
            print(f"image {i-1} and {i} have not enough matches - {len(good_matches)}/{MIN_MATCH_COUNT}")

    # ========== RANSAC处理匹配结果 ==========
    if len(good_matches_list) > 0:
        # 存储几何模型信息
        geometric_models = []

        for i in range(len(good_matches_list)):
            # 使用改进的RANSAC函数同时估计F和H矩阵
            model_type, matrix, inliers, num_inliers = ransac_geometric_consistency(
                good_matches_list[i], 
                keypoints_list[i], 
                keypoints_list[i+1],
                ransac_threshold=2,
                max_iterations=100,
                confidence=0.99
            )
            
            geometric_models.append({
                'model_type': model_type,
                'matrix': matrix,
                'inliers': inliers,
                'num_inliers': num_inliers,
                'img_idx1': i,
                'img_idx2': i+1
            })
            
            print(f"图像 {i} 和 {i+1} 之间的最佳模型: {model_type}, 内点数: {num_inliers}")
        
        # 以第一对图像为例进行可视化
        if len(good_matches_list) > 0:
            first_model = geometric_models[0]
            img1 = imgs_list[first_model['img_idx1']]
            img2 = imgs_list[first_model['img_idx2']]
            kp1 = keypoints_list[first_model['img_idx1']]
            kp2 = keypoints_list[first_model['img_idx2']]
            good_matches = good_matches_list[0]
            inliers = first_model['inliers']

            # 创建拼接图像
            nHeight = max(img1.size[1], img2.size[1])
            nWidth = img1.size[0] + img2.size[0]
            newimg = np.zeros((nHeight, nWidth, 3), dtype=np.uint8)
            
            # 将灰度图像转换为彩色图像
            img1_np = np.array(img1)
            img2_np = np.array(img2)
            
            if len(img1_np.shape) == 2:
                img1_np = cv2.cvtColor(img1_np, cv2.COLOR_GRAY2BGR)
            if len(img2_np.shape) == 2:
                img2_np = cv2.cvtColor(img2_np, cv2.COLOR_GRAY2BGR)
                
            # 放置图像
            newimg[:img1.size[1], :img1.size[0]] = img1_np
            newimg[:img2.size[1], img1.size[0]:img1.size[0]+img2.size[0]] = img2_np
            
            # 提取匹配点的坐标
            src_pts = np.float32([[kp1[m.queryIdx]['x'], kp1[m.queryIdx]['y']] for m in inliers]).reshape(-1, 1, 2)
            dst_pts = np.float32([[kp2[m.trainIdx]['x'], kp2[m.trainIdx]['y']] for m in inliers]).reshape(-1, 1, 2)
            
            # 假设已知相机内参K (这里需要真实的相机内参)
            # 这里使用一个示例内参，实际应用中应该使用真实的相机内参
            K = np.array([
                [1000, 0, img1.size[0]/2],
                [0, 1000, img1.size[1]/2],
                [0, 0, 1]
            ])
            # 根据模型类型进行不同的可视化
            if first_model['model_type'] == 'H':
                # 使用decompose_homography函数从单应矩阵H分解得到R和T
                H = first_model['matrix']
                best_R, best_t, best_normal = decompose_homography(H, K)
            elif first_model['model_type'] == 'F':
                # 基础矩阵 - 绘制极线
                F = first_model['matrix']
                # 使用decompose_fundamental_matrix函数从基础矩阵F分解得到R和T
                R, t, mask = decompose_fundamental_matrix(F, K, src_pts, dst_pts)   
            

            # 绘制匹配线 - 只绘制内点
            for m in inliers:
                pt1 = (int(kp1[m.queryIdx]['x']), int(kp1[m.queryIdx]['y']))
                pt2 = (int(kp2[m.trainIdx]['x']) + img1.size[0], int(kp2[m.trainIdx]['y']))
                cv2.line(newimg, pt1, pt2, (0, 0, 255), 1)  # 红色线条
            
            # 显示结果
            plt.figure(figsize=(12, 10))
            plt.imshow(cv2.cvtColor(newimg, cv2.COLOR_BGR2RGB))
            plt.title(f"model type: {first_model['model_type']}, num of inliers: {first_model['num_inliers']}")
            plt.axis('off')
            plt.show(block=True)

        else:
            print("没有找到足够的匹配点")

def display_exif_info(img_data):
    """显示重要的EXIF信息"""
    exif = img_data['exif']
    if not exif:
        print("无EXIF信息")
        return
    
    print("\nEXIF信息:")
    # 显示关键EXIF字段
    keys_to_display = ['DateTime', 'Make', 'Model', 'ExposureTime', 
                       'FNumber', 'ISOSpeedRatings', 'FocalLength']
    
    for key in keys_to_display:
        if key in exif:
            print(f"  {key}: {exif[key]}")
    
    # 显示其他EXIF字段数量
    other_keys = [k for k in exif.keys() if k not in keys_to_display]
    if other_keys:
        print(f"  其他 {len(other_keys)} 个EXIF字段")

def display_gps_info(img_data):
    """显示GPS信息（如果存在）"""
    gps_info = img_data['gps']
    if not gps_info:
        print("无GPS信息")
        return
    
    print("\nGPS信息:")
    # 显示关键GPS字段
    keys_to_display = ['GPSLatitude', 'GPSLongitude', 'GPSAltitude', 
                       'GPSDateStamp', 'GPSTimeStamp']
    
    for key in keys_to_display:
        if key in gps_info:
            print(f"  {key}: {gps_info[key]}")
    
    # 显示其他GPS字段数量
    other_keys = [k for k in gps_info.keys() if k not in keys_to_display]
    if other_keys:
        print(f"  其他 {len(other_keys)} 个GPS字段")

if __name__ == "__main__":
    main()