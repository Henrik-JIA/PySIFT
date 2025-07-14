import sys
import os
import logging
import numpy as np
import cv2
from src.util.image_loader import ImageLoader
from src.Pyramid.extract_sift_features import extract_sift_features
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

def match_descriptors(descriptors1, descriptors2, ratio_threshold=0.7):
    """
    使用FLANN（快速近似最近邻搜索库）匹配器匹配两组描述符，并应用Lowe's ratio test
    
    参数:
        descriptors1 (numpy.ndarray): 第一组描述符
        descriptors2 (numpy.ndarray): 第二组描述符
        ratio_threshold (float): Lowe's ratio test的阈值（默认0.7）
    
    返回:
        list: 筛选后的匹配点列表
    """
    # 初始化FLANN匹配器
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # 进行KNN匹配
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # 应用Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    
    return good_matches

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
        print(f"\n图片 #{idx+1}: {img_exif['path']}")
        print(f"尺寸: {img_exif['width']}x{img_exif['height']}")
        print(f"格式: {img_exif['format']}")
        print(f"大小: {img_exif['size'] / 1024:.2f} KB")
        # 显示EXIF信息
        display_exif_info(img_exif)
        # 显示GPS信息
        display_gps_info(img_exif)

        # 打开图片并转为灰度图
        image = Image.open(img_exif['path']).convert('L')
        
        imgs_list.append(image)

        # 提取SIFT特征
        keypoints, descriptors = extract_sift_features(image, idx)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    good_matches = match_descriptors(descriptors_list[0], descriptors_list[1], ratio_threshold=0.7)

    if len(good_matches) > MIN_MATCH_COUNT:
        kp1 = keypoints_list[0]
        kp2 = keypoints_list[1]
        img1 = imgs_list[0]
        img2 = imgs_list[1]

        # 提取匹配点的坐标（针对字典结构的关键点）
        src_pts = np.float32([ [kp1[m.queryIdx]['x'], kp1[m.queryIdx]['y'] ] for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([ [kp2[m.trainIdx]['x'], kp2[m.trainIdx]['y'] ] for m in good_matches]).reshape(-1, 1, 2)
        
        # 计算单应性矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            # 在场景图像上绘制检测到的模板边界
            w, h = img1.size
            pts = np.float32([[0, 0],
                              [0, h - 1],
                              [w - 1, h - 1],
                              [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            
            # 绘制多边形边界
            img2_np = np.array(img2)
            img2_with_poly = cv2.polylines(img2_np, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            
            # 创建拼接图像
            nHeight = max(img1.size[1], img2.size[1]) # 高度是size[1]
            nWidth = img1.size[0] + img2.size[0] # 宽度是size[0]
            newimg = np.zeros((nHeight, nWidth, 3), dtype=np.uint8)
            
            # 放置图像 - 需要将灰度图像转换为彩色图像
            img1_np = np.array(img1)
            img2_np = np.array(img2_with_poly)
            
            # 如果是灰度图像，转换为3通道
            if len(img1_np.shape) == 2:
                img1_np = cv2.cvtColor(img1_np, cv2.COLOR_GRAY2BGR)
            if len(img2_np.shape) == 2:
                img2_np = cv2.cvtColor(img2_np, cv2.COLOR_GRAY2BGR)
                
            # 放置图像
            newimg[:img1.size[1], :img1.size[0]] = img1_np
            newimg[:img2.size[1], img1.size[0]:img1.size[0]+img2.size[0]] = img2_np
            
            # 绘制匹配线
            for m in good_matches:
                pt1 = (int(kp1[m.queryIdx]['x']), int(kp1[m.queryIdx]['y']))
                pt2 = (int(kp2[m.trainIdx]['x']) + img1.size[0], int(kp2[m.trainIdx]['y']))
                cv2.line(newimg, pt1, pt2, (0, 0, 255), 1)  # 红色线条
            
            # 显示结果
            plt.figure(figsize=(15, 10))
            plt.imshow(cv2.cvtColor(newimg, cv2.COLOR_BGR2RGB))  # 转换为RGB以正确显示颜色
            plt.title(f"Matching points: {len(good_matches)}")
            plt.axis('off')
            plt.show(block=True)
        else:
            print("Homography matrix calculation failed")
    else:
        print("Not enough matches are found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT))

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