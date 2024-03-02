import cv2
import numpy as np
imgname_01 = 'X.png'
imgname_02 = 'A.png'
imgname_03 = 'B.png'
imgname_04 = 'C.png'

orb = cv2.ORB_create(3500)#生成的特征点数目上限，过低有匹配失败风险，过高有产生误匹配的风险且速度较慢

imgX = cv2.imread(imgname_01)
imgA = cv2.imread(imgname_02)
imgB = cv2.imread(imgname_03)
imgC = cv2.imread(imgname_04)
kpX, desX = orb.detectAndCompute(imgX, None)
kpA, desA = orb.detectAndCompute(imgA, None)
kpB, desB = orb.detectAndCompute(imgB, None)
kpC, desC = orb.detectAndCompute(imgC, None)#当需要添加更多零部件图像时，依次仿写即可
# 提取并计算特征点
bf = cv2.BFMatcher()
# knn筛选结果
matchesA = bf.knnMatch(desA, trainDescriptors=desX, k=2)
matchesB = bf.knnMatch(desB, trainDescriptors=desX, k=2)
matchesC = bf.knnMatch(desC, trainDescriptors=desX, k=2)


goodA = []
for m, n in matchesA:
    if m.distance < 0.75 * n.distance:
        goodA.append(m)
min_match_count = 10

if len(goodA) > 10:
    src_pts = np.float32([kpA[m.queryIdx].pt for m in goodA]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpX[m.trainIdx].pt for m in goodA]).reshape(-1, 1, 2)
    ransacReprojThreshold = 8.0
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold)
    matchesMask = mask.ravel().tolist()
    h, w, mode = imgA.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
else:
    matchesMask = None

goodB = []
for m, n in matchesB:
    if m.distance < 0.75 * n.distance:
        goodB.append(m)
min_match_count = 10

if len(goodB) > 10:
    src_pts = np.float32([kpB[m.queryIdx].pt for m in goodB]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpX[m.trainIdx].pt for m in goodB]).reshape(-1, 1, 2)
    ransacReprojThreshold = 8.0
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold)
    matchesMask = mask.ravel().tolist()
    h, w, mode = imgB.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
else:
    matchesMask = None

goodC = []
for m, n in matchesC:
    if m.distance < 0.75 * n.distance:
        goodC.append(m)
min_match_count = 10

if len(goodC) > 10:
    src_pts = np.float32([kpC[m.queryIdx].pt for m in goodC]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpX[m.trainIdx].pt for m in goodC]).reshape(-1, 1, 2)
    ransacReprojThreshold = 8.0
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold)
    matchesMask = mask.ravel().tolist()
    h, w, mode = imgC.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
else:
    matchesMask = None

A=len(goodA)
B=len(goodB)
C=len(goodC)
lister = [A, B, C]
list_of_len = ["A", "B", "C"]
# 高指数列表，如果有两个相等则可以有多个元素
high = [i for i, x in enumerate(lister) if x == max(lister)]

# 创建匹配程度最高的列表
closest = [list_of_len[i] for i in high]

# 输出
print("The closet image is ", ", ".join(closest), "，and its number of feature points is", max(lister))
