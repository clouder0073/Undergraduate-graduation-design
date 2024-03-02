import cv2

img1 = cv2.imread('0.png', 0)
imgA = cv2.imread('A.png', 0)
imgB = cv2.imread('B.png', 0)
imgC = cv2.imread('C.png', 0)
imgD = cv2.imread('D.png', 0)
imgE = cv2.imread('E.png', 0)

# 最大特征点数。
orb = cv2.ORB_create(3500)

kp1, des1 = orb.detectAndCompute(img1, None)
kpA, desA = orb.detectAndCompute(imgA, None)
kpB, desB = orb.detectAndCompute(imgB, None)
kpC, desC = orb.detectAndCompute(imgC, None)
kpD, desD = orb.detectAndCompute(imgD, None)
kpE, desE = orb.detectAndCompute(imgE, None)

# 提取并计算特征点
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# knn筛选结果
matchesA = bf.knnMatch(des1, trainDescriptors=desA, k=2)
matchesB = bf.knnMatch(des1, trainDescriptors=desB, k=2)
matchesC = bf.knnMatch(des1, trainDescriptors=desC, k=2)
matchesD = bf.knnMatch(des1, trainDescriptors=desD, k=2)
matchesE = bf.knnMatch(des1, trainDescriptors=desE, k=2)
A = len([m for (m, n) in matchesA if m.distance < 0.75 * n.distance])
B = len([m for (m, n) in matchesB if m.distance < 0.75 * n.distance])
C = len([m for (m, n) in matchesC if m.distance < 0.75 * n.distance])
D = len([m for (m, n) in matchesD if m.distance < 0.75 * n.distance])
E = len([m for (m, n) in matchesE if m.distance < 0.75 * n.distance])
lister = [A, B, C, D, E]
list_of_len = ["A", "B", "C", "D", "E"]
# 高指数列表，如果有两个相等则可以有多个元素
high = [i for i, x in enumerate(lister) if x == max(lister)]

# 创建匹配程度最高的列表
closest = [list_of_len[i] for i in high]

# 输出
print("最接近的图片是", ", ".join(closest), "，它的特征点数目是", max(lister))
# print(A,B,C,D,E)