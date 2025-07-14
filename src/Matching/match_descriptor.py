import numpy as np
import cv2

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
