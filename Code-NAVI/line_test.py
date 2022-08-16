import cv2
import scipy.stats
import numpy as np
from matplotlib import pyplot as plt
def edge_detection(img_path):
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(binary, 100, 200)
    plt.imshow(binary, cmap='gray')
    plt.show()


    indices = np.where(edges != [0])
    coordinates = zip(indices[1], indices[0])
    line = np.column_stack((indices[1], indices[0]))
    for point in coordinates:
        cv2.circle(img,tuple(point), 3, (0,0,255))
    img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))  # Resize image



    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(line)
    r_value = abs(r_value)
    print('Line r_value : {}'.format(r_value))

    cv2.imshow("img", img)
    cv2.waitKey(0)

    return r_value

def line_detection(img_path):
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    plt.imshow(binary, cmap='gray')
    plt.show()
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    contours = contours[-1].squeeze(1)
    contour = cv2.approxPolyDP(contours, 15, True)
    #
    for point in contours:
        cv2.circle(img,tuple(point),3,(0,0,255))

    img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))  # Resize image
    cv2.imshow("img", img)
    cv2.waitKey(0)

    vertice_set = contour.squeeze(1)
    vertice_sum = np.sum(vertice_set, axis=1)
    tl = np.argmin(vertice_sum)
    lr = np.argmax(vertice_sum)

    index = [0, 1, 2, 3]
    index.remove(tl)
    index.remove(lr)
    if vertice_set[index[0]][0] > vertice_set[index[0]][1]:    # if x > y, tr
        tr = index[0]
        ll = index[1]
    else:
        tr = index[1]
        ll = index[0]

    # vertices_id = [tl, tr, ll, lr]
    vertices_id = [tl, ll, lr, tr]

    vertice_rec = []
    for vertex_id in vertices_id:
        for i in range(contours.shape[0]):
            if np.array_equal(contours[i], vertice_set[vertex_id]):
                vertice_rec.append(i)
                break


    print('total points : ', contours.shape[0])
    line = []
    line.append(contours[vertice_rec[0]:vertice_rec[1]+1])
    line.append(contours[vertice_rec[1]:vertice_rec[2]+1])
    line.append(contours[vertice_rec[2]:vertice_rec[3]+1])
    line_4_a = contours[:vertice_rec[0]+1]
    line_4_b = contours[vertice_rec[3]:]
    line_4_ab = np.concatenate((line_4_a, line_4_b), axis=0)
    line.append(line_4_ab)

    r_sq = []
    for i in range(len(line)):

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(line[i])
        r_sq.append(r_value ** 2)
        print('Line {} r_value : {:.4f}'.format(i, r_sq[i]))


    # for point in line[-1]:
    #     cv2.circle(img,tuple(point),3,(0,0,255))



    # cv2.drawContours(img, o_contours, -1, (0, 0, 255), 1)
    # cv2.circle(img,tuple(contours[vertice_rec[0]]),3,(0,0,255))
    # cv2.circle(img,tuple(contours[vertice_rec[1]]),3,(0,0,255))
    # cv2.circle(img,tuple(contours[vertice_rec[2]]),3,(0,0,255))
    # cv2.circle(img,tuple(contours[vertice_rec[3]]),3,(0,0,255))


    # cv2.circle(img,tuple(vertice_set[0]),3,(0,255,255))
    # cv2.circle(img,tuple(vertice_set[1]),3,(0,255,255))
    # cv2.circle(img,tuple(vertice_set[2]),3,(0,255,255))
    # cv2.circle(img,tuple(vertice_set[3]),3,(0,255,255))


    # img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))  # Resize image

    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    return r_sq



for k in range(1):
    # before_err = line_detection('D:/Download/Line test image/after correction1/c/before correction{}c.bmp'.format(k))
    # after_err = line_detection('D:/Download/Line test image/after correction1/c/after correction{}c.bmp'.format(k))
    k = 0

    # before_err = edge_detection('./Undistortion Test 20220408/Undistortion Test 20220408/AUX273/before correction{}.bmp'.format(k))
    # after_err = edge_detection('./Undistortion Test 20220408/Undistortion Test 20220408/AUX273/after correction{}.bmp'.format(k))

    # before_err = edge_detection('./Undistortion Test 20220408/Undistortion Test 20220408/Live streamer 513/before correction{}.bmp'.format(k))
    # after_err = edge_detection('./Undistortion Test 20220408/Undistortion Test 20220408/Live streamer 513/after correction{}.bmp'.format(k))

    # improvement = (after_err - before_err) / before_err * 100
    # print('Image{} improve {:.2f}%'.format(k, improvement))

    # before_err = line_detection('./Undistortion Test 20220408/Undistortion Test 20220408/AUX273/before correction{}.bmp'.format(k))
    # after_err = line_detection('./Undistortion Test 20220408/Undistortion Test 20220408/AUX273/after correction{}.bmp'.format(k))

    before_err = line_detection('./Undistortion Test 20220408/Undistortion Test 20220408/Live streamer 513/before correction{}.bmp'.format(k))
    after_err = line_detection('./Undistortion Test 20220408/Undistortion Test 20220408/Live streamer 513/after correction{}.bmp'.format(k))
    for i in range(len(before_err)):
        improvement = (after_err[i] - before_err[i]) / before_err[i] * 100
        print('Image{} line{} improve {:.2f}%'.format(k, i, improvement))


# 3/25 想以r square作為判斷直線的程度
# 演算法找到一整條完整的縣還沒寫好
# 現在都是手動輸入