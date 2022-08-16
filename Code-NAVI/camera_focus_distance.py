import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import time
def TENG_plot(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gaussianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gaussianY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    absX = cv2.convertScaleAbs(gaussianX)
    absY = cv2.convertScaleAbs(gaussianX)
    print(np.mean(gaussianX * gaussianX + gaussianY * gaussianY))

    # 將兩個軸向的測邊結果相加，形成完整輪廓
    dst = cv2.addWeighted(absX, 0.5, absX, 0.5, 0)
    plt.subplot(111)
    plt.imshow(dst, cmap='gray')
    plt.title('Complete result')
    plt.xticks([])
    plt.yticks([])

    plt.show()
    return dst

    # plt.savefig('./FocusImage/Sobel_operator_result_{}.jpg'.format(27))




def TENG(img):

    gaussianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gaussianY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    absX = cv2.convertScaleAbs(gaussianX)
    absY = cv2.convertScaleAbs(gaussianX)
    # return np.mean(gaussianX * gaussianX + gaussianY * gaussianY)
    return np.mean(absX + absY)

def LAPM(img):

    kernel = np.array([-1, 2, -1])
    laplacianX = np.abs(cv2.filter2D(img, -1, kernel))
    laplacianY = np.abs(cv2.filter2D(img, -1, kernel.T))
    return np.mean(laplacianX + laplacianY)

def LAPV(img):

    return np.std(cv2.Laplacian(img, cv2.CV_64F)) ** 2


def distance_test():

    path = "C:/Users/Romeo/Downloads/depth image2"
    imgs = sorted(glob.glob(os.path.join(path, '*.bmp')))

    x_points = np.arange(1, len(imgs)+1)
    y_points_1 = np.zeros_like(x_points, dtype=float)
    y_points_2 = np.zeros_like(x_points, dtype=float)
    y_points_3 = np.zeros_like(x_points, dtype=float)

    for i, img in enumerate(imgs):
        img_array = cv2.imread(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        y_points_1[i] = TENG(gray)
        y_points_2[i] = LAPM(gray)
        y_points_3[i] = LAPV(gray)


        print("Image distance {} : {:.2f}   {:.2f}   {:.2f}".format(os.path.basename(img), y_points_1[i], y_points_2[i], y_points_3[i]))

    plt.plot(x_points, y_points_1)
    plt.xticks(x_points, rotation=45)
    plt.title('Method : TENG')
    plt.grid()
    plt.show()

    plt.plot(x_points, y_points_2)
    plt.xticks(x_points, rotation=45)
    plt.title('Method : LAPM')
    plt.grid()
    plt.show()

    plt.plot(x_points, y_points_3)
    plt.xticks(x_points, rotation=45)
    plt.title('Method : LAPV')
    plt.grid()
    plt.show()

def fps_measure():
    path = "C:/Users/Romeo/Downloads/depth image2"
    imgs = sorted(glob.glob(os.path.join(path, '*.bmp')))

    start = time.time()

    for i, img in enumerate(imgs):
        img_array = cv2.imread(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        value = TENG(gray)

    end = time.time()

    fps = len(imgs) / (end - start)
    print(fps)

if __name__ == '__main__':

    # fps_measure()
    #
    # path = "C:/Users/Romeo/Downloads/depth image/img_06.bmp"
    # TENG(path)
    # TENG_plot(path)
    # plt.imshow(diff, cmap='gray')
    # plt.title('Diff')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    distance_test()

