from matplotlib import pyplot as plt
import cv2
#from skimage.metrics import structural_similarity


def compare(result1, result2, img0, i):  # 定义灰度直方图比较函数
    if result2 > 0.99 and result1 < 0.03:  # 相关性大于0.99且巴氏距离小于0.03为无故障，反之为有故障
        print("Faultless")
    else:
        print("Faulty")


def create_hist(img):  # 定义绘制直方图函数
    img = cv2.imread(img)  # 读取图片
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片转化为8bit灰度图

    plt.imshow(img_gray, cmap=plt.cm.gray)  # 显示图片
    hist = cv2.calcHist([img], [0], None, [255], [0, 255])  # 灰度直方图
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 255])
    plt.show()
    return hist


hist1 = create_hist("0.png")  # 给标准样品绘制直方图
for i in range(1, 2):  #range函数: #1: range(10),等于[0，1，2，3，4，5，6，7，8，9]
    # 2: range(1,10),等于[1，2，3，4，5，6，7，8，9]
    # 3: range(1,10,2),等于[1，3，5，7，9]  此处只执行了一次，当需要对多个图像进行故障检测时，改变range函数中的值即可
    print(i)  # 打印图片序号
    img = cv2.imread("%d.png" % (i), 1)#读取文件夹中名为i.png的图片
    hist2 = create_hist("%d.png" % (i))  # 给测试样品绘制直方图
    match1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)  # 返回巴氏距离
    match2 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)  # 返回相关性
    print("巴氏距离：%s, 相关性：%s" % (match1, match2))
    print("\n")
    compare(match1, match2, img, i)
