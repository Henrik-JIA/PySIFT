import numpy as np
import cv2

def ransac_geometric_consistency(matches, keypoints1, keypoints2, ransac_threshold=3.0, max_iterations=30, confidence=0.99):
    """
    使用RANSAC算法同时估计基础矩阵F和单应矩阵H，并选择得分更高的矩阵进行几何一致性过滤
    
    参数:
        matches: match_descriptors函数返回的匹配点列表
        keypoints1: 第一幅图像的关键点(字典格式)
        keypoints2: 第二幅图像的关键点(字典格式)
        ransac_threshold: RANSAC算法的阈值(像素单位)
        max_iterations: RANSAC算法的最大迭代次数
        confidence: RANSAC算法的置信度
        
    返回:
        best_model: 'F'或'H'，表示最佳模型类型
        best_matrix: 最佳模型矩阵(基础矩阵F或单应矩阵H)
        inliers_matches: 满足最佳模型的内点匹配列表
        num_inliers: 内点数量
    """
    # 获取匹配点的坐标(针对字典结构的关键点)
    points1 = np.float32([[keypoints1[m.queryIdx]['x'], keypoints1[m.queryIdx]['y']] for m in matches])
    points2 = np.float32([[keypoints2[m.trainIdx]['x'], keypoints2[m.trainIdx]['y']] for m in matches])
    
    # 重塑点对格式用于单应矩阵计算
    src_pts = points1.reshape(-1, 1, 2)
    dst_pts = points2.reshape(-1, 1, 2)
    
    # 1. 使用RANSAC算法计算基础矩阵F
    F, F_mask = cv2.findFundamentalMat(
        points1, points2, 
        method=cv2.FM_RANSAC, 
        ransacReprojThreshold=ransac_threshold,
        confidence=confidence,
        maxIters=max_iterations
    )
    
    # 2. 使用RANSAC算法计算单应矩阵H
    H, H_mask = cv2.findHomography(
        src_pts, dst_pts, 
        method=cv2.RANSAC, 
        ransacReprojThreshold=ransac_threshold,
        maxIters=max_iterations,
        confidence=confidence
    )
    
    # 3. 计算F矩阵和H矩阵的内点数量
    if F_mask is not None:
        F_inliers_mask = F_mask.ravel().astype(bool)
        F_num_inliers = np.sum(F_inliers_mask)
        F_inliers_matches = [matches[i] for i in range(len(matches)) if F_inliers_mask[i]]
    else:
        F_num_inliers = 0
        F_inliers_matches = []
    
    if H_mask is not None:
        H_inliers_mask = H_mask.ravel().astype(bool)
        H_num_inliers = np.sum(H_inliers_mask)
        H_inliers_matches = [matches[i] for i in range(len(matches)) if H_inliers_mask[i]]
    else:
        H_num_inliers = 0
        H_inliers_matches = []
    
    # 4. 比较F矩阵和H矩阵的内点数量，选择得分更高的矩阵
    if F_num_inliers >= H_num_inliers:
        return 'F', F, F_inliers_matches, F_num_inliers
    else:
        return 'H', H, H_inliers_matches, H_num_inliers
    
def decompose_homography(H, K):
    """
    从单应矩阵H分解出相机的旋转矩阵R和平移向量t
    
    参数:
        H: 单应矩阵
        K: 相机内参矩阵
        
    返回:
        best_R: 最佳旋转矩阵
        best_t: 最佳平移向量
        best_normal: 平面法向量
    """
    # 使用OpenCV函数分解单应矩阵
    num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
    
    # 打印所有可能的解
    print(f"单应矩阵分解得到 {num_solutions} 个可能解")
    for i in range(num_solutions):
        print(f"解 #{i+1}:")
        print("旋转矩阵R:")
        print(rotations[i])
        print("平移向量t:")
        print(translations[i])
        print("平面法向量n:")
        print(normals[i])
        print("-------------------")
    
    # 选择最合理的解
    # 通常需要额外的约束条件来选择正确的解
    # 这里简单地选择第一个解作为结果
    best_R = rotations[0]
    best_t = translations[0]
    best_normal = normals[0]
    
    # 可以添加更复杂的选择逻辑，例如：
    # 1. 检查重投影误差
    # 2. 确保所有三维点都在相机前方（正深度约束）
    # 3. 利用先验知识（如相机运动方向）
    
    return best_R, best_t, best_normal

def decompose_fundamental_matrix(F, K, points1, points2):
    """
    从基础矩阵F分解出相机的旋转矩阵R和平移向量t
    
    参数:
        F: 基础矩阵
        K: 相机内参矩阵
        points1: 第一幅图像中的点坐标 (Nx2数组)
        points2: 第二幅图像中的点坐标 (Nx2数组)
        
    返回:
        R: 旋转矩阵
        t: 平移向量
        mask: 三角化点的掩码，指示哪些点成功三角化
    """
    # 从基础矩阵计算本质矩阵
    E = K.T @ F @ K
    
    # 从本质矩阵恢复旋转矩阵和平移向量
    # recoverPose返回：点数，R，t，掩码
    point_count, R, t, mask = cv2.recoverPose(E, points1, points2, K)
    
    print("从基础矩阵分解:")
    print(f"成功三角化的点数: {point_count}")
    print("旋转矩阵R:")
    print(R)
    print("平移向量t:")
    print(t)
    
    return R, t, mask
