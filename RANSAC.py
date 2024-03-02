import numpy as np
import cv2
from matplotlib import pyplot as plt

imgname_01 = 'E.png'
imgname_02 = '0.png'
orb = cv2.ORB_create(2000)

img_01 = cv2.imread(imgname_01)
img_02 = cv2.imread(imgname_02)

print(img_01.shape)

keypoint_01, descriptor_01 = orb.detectAndCompute(img_01, None)
keypoint_02, descriptor_02 = orb.detectAndCompute(img_02, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptor_01, descriptor_02, k=2)
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

    # 如果找到了足够的匹配，就提取两幅图像中匹配点的坐标，把它们传入到函数中做变换
print(type(good))

min_match_count = 10
if len(good) > 10:
    src_pts = np.float32([keypoint_01[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoint_02[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    print(src_pts.shape)
    print(dst_pts.shape)

    '''
    cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold) 利用RANSAC方法计算单应矩阵
    ransacReprojThreshold为阈值，当某一个匹配与估计的假设小于阈值时，则被认为是一个内点，默认是3
    返回值：M 和 mask
    mask：标记矩阵，标记内点和外点 他和m1，m2的长度一样，当一个m1和m2中的点为内点时，mask相应的标记为1，反之为0
    '''
    ransacReprojThreshold = 8.0
    # 返回值中 M 为单应性矩阵。
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold)
    print(mask.shape)
    matchesMask = mask.ravel().tolist()#将多维数组转换为一维数组
    h, w, mode = img_01.shape#获取原图像的高和宽
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)#使用得到的变换矩阵对原图像的四个变换获得在目标图像上的坐标
    dst = cv2.perspectiveTransform(pts, M)#透视变换函数cv2.perspectiveTransform: 输入的参数是两种数组，并返回dst矩阵——扭转矩阵
else:
    print('Can not  matches!')
    matchesMask = None

img3 = cv2.drawMatches(img_01, keypoint_01, img_02, keypoint_02, good, None)

img_ransac = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)  # RGB
plt.imshow(img_ransac)
plt.savefig('img_ORB_by_RANASC.png')

cv2.destroyAllWindows()
