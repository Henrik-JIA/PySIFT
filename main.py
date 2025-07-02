import sys
import os
import numpy as np
from src.util.image_loader import ImageLoader
from src.Pyramid.extract_sift_features import extract_sift_features
from PIL import Image  
import matplotlib.pyplot as plt  

def main():
    """
    主函数：加载并处理图片数据
    """
    # 默认使用项目中的data文件夹
    data_path = "data"
    
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
    keypoints_list = []
    descriptors_list = []
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
        
        # 提取SIFT特征
        keypoints, descriptors = extract_sift_features(image)
        print(keypoints)
        print(descriptors)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)


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