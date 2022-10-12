import numpy as np
import math
from scipy.linalg import null_space
# import cv2.aruco as aruco
import cv2
from needle_utils_temp import camera_para_retrieve
def scale_estimation(q1, q2, q3, d1, d2, mtx):

    x_scale = (3.45 / 1000)  # pixel size : 3.45 Micrometer = 0.00345 mm
    y_scale = (3.45 / 1000)
    # xp = mtx[0][2] * x_scale
    # yp = mtx[1][2] * y_scale
    f = 8

    scale = np.array([x_scale, y_scale, 0])
    trans = np.array([mtx[0][2], mtx[1][2], 0])

    q1 -= trans
    q2 -= trans
    q3 -= trans

    q1 *= scale
    q2 *= scale
    q3 *= scale


    F = np.array([0, 0, f])

    v1 = (F - q1) / np.linalg.norm((F - q1), axis=0)
    v2 = (F - q2) / np.linalg.norm((F - q2), axis=0)
    v3 = (F - q3) / np.linalg.norm((F - q3), axis=0)

    A = v1
    n = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2), axis=0)
    B = np.cross(n, A)

    a1 = np.dot(v1, A)
    a2 = np.dot(v2, A)
    a3 = np.dot(v3, A)

    b1 = np.dot(v1, B)
    b2 = np.dot(v2, B)
    b3 = np.dot(v3, B)

    S = np.array([[d2*a1, -d2*a2-d1*a2, d1*a3], [d2*b1, -d1*b2-d2*b2, d1*b3]])
    ns = null_space(S)

    #check

    d1p = np.linalg.norm(ns[0] * v1 - ns[1] * v2, axis=0)
    d2p = np.linalg.norm(ns[1] * v2 - ns[2] * v3, axis=0)
    scale_1 = d1 / d1p
    scale_2 = d2 / d2p
    s1 = ns[0] * scale_1
    s2 = ns[1] * scale_1
    s3 = ns[2] * scale_1

    c1 = v1 * s1
    c2 = v2 * s1
    c3 = v3 * s1

    p1 = 0.1 * (F + v1 * s1) * np.array([-1, -1, 1])
    p2 = 0.1 * (F + v2 * s2) * np.array([-1, -1, 1])
    p3 = 0.1 * (F + v3 * s3) * np.array([-1, -1, 1])




    g1 = np.linalg.norm((p1 - p2), axis=0)
    g2 = np.linalg.norm((p2 - p3), axis=0)


    # print("{}   {}   {}".format(p1 * 0.1, p2 *0.1, p3*0.1))

    unit = (p2 - p1) / np.linalg.norm((p2 - p1), axis=0)
    # le = p1 + unit * 21
    # print(le)
    # print('p', int(p1[2]), int(p2[2]), int(p3[2]), int(le[2]))

    return p1

def aruco(frame, mtx, dist):

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    squareLength = 1.67
    markerLength = 0.9
    arucoParams = cv2.aruco.DetectorParameters_create()

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=arucoParams)
    if np.any(ids != None):
        for i in range(len(ids)):

            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], markerLength, mtx, dist)

            # frame = cv2.aruco.drawAxis(frame, mtx, dist, rvec, tvec, 1)  # Draw axis

            print(ids[i], rvec, tvec)

    # cv2.imshow('frame', frame)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return tvec


def undistorted():
    img = cv2.imread('./needle_detect_Img/2022-07-18_11-35-08.jpg')
    mtx, dist = camera_para_retrieve()
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undist = cv2.undistort(img, mtx, dist, None, newcameramtx)
    cv2.imshow('undist', undist)
    cv2.waitKey(0)
    return undist

def image_test0():
    frame = cv2.imread('./linearImage/3.bmp')
    mtx = np.array([[2433.68459290071, 0, 718.87372815422], [0, 2440.24567073231, 601.529413632018], [0, 0, 1]])
    dist = np.array([[-0.0816183865808583], [-0.0605561177723738], [0.0019214883496733], [-0.00142703885621291], [1.20797639940181]])

    tvec = aruco(frame, mtx, dist)
    d1, d2 = 32, 40

    q1 = np.array([969, 748, 0], dtype='float64')
    q2 = np.array([904, 736, 0], dtype='float64')
    q3 = np.array([822, 722, 0], dtype='float64')
    est_tvec = scale_estimation(q1, q2, q3, d1, d2, mtx, dist )
    err = abs((tvec - est_tvec))
    print(err)

def image_test1():
    frame = cv2.imread('./linearImage/10.bmp')
    mtx = np.array([[2433.68459290071, 0, 718.87372815422], [0, 2440.24567073231, 601.529413632018], [0, 0, 1]])
    dist = np.array([[-0.0816183865808583], [-0.0605561177723738], [0.0019214883496733], [-0.00142703885621291], [1.20797639940181]])

    d1, d2 = 32, 40

    q1 = np.array([587, 491, 0], dtype='float64')
    q2 = np.array([631, 455, 0], dtype='float64')
    q3 = np.array([688, 413, 0], dtype='float64')
    est_tvec = scale_estimation(q1, q2, q3, d1, d2, mtx, dist)

    # tvec = aruco(frame, mtx, dist)
    # err = abs((tvec - est_tvec))
    # print(err)

def image_test2():
    frame = cv2.imread('./needle_detect_Img/2022-07-18_11-35-08.jpg')
    mtx, dist = camera_para_retrieve()
    tvec = aruco(frame, mtx, dist)
    d1, d2 = 20, 10

    q1 = np.array([582, 492, 0], dtype='float64')
    q2 = np.array([539, 448, 0], dtype='float64')
    q3 = np.array([515, 426, 0], dtype='float64')

    est_tvec = scale_estimation(q1, q2, q3, d1, d2, mtx, dist)
    err = abs((tvec - est_tvec))
    print(err)

def image_test3():
    frame = cv2.imread('./needle_detect_Img/2022-07-18_11-35-08.jpg')
    mtx, dist = camera_para_retrieve()
    tvec = aruco(frame, mtx, dist)
    d1, d2 = 10, 40

    # q1 = np.array([692, 611, 0], dtype='float64')
    # q2 = np.array([603, 515, 0], dtype='float64')
    # q3 = np.array([581, 493, 0], dtype='float64')

    q1 = np.array([603, 515, 0], dtype='float64')
    q2 = np.array([581, 493, 0], dtype='float64')
    q3 = np.array([496, 401, 0], dtype='float64')

    est_tvec = scale_estimation(q1, q2, q3, d1, d2, mtx, dist)
    err = abs((tvec - est_tvec))
    print(err)

def image_test4():
    frame = cv2.imread('./needle_detect_Img/2022-07-18_11-35-08.jpg')
    mtx, dist = camera_para_retrieve()
    tvec = aruco(frame, mtx, dist)

    coords = np.load('coordinate.npy')
    test_time = 20
    for i in range(test_time):
        choices = np.random.choice(17, 3, replace=False)
        choices = np.sort(choices)
        q1 = coords[choices[0]]
        q2 = coords[choices[1]]
        q3 = coords[choices[2]]
        if choices[2] != 16:
            d1 = (choices[1] - choices[0]) * 10
            d2 = (choices[2] - choices[1]) * 10
        else:
            d1 = (choices[1] - choices[0]) * 10
            d2 = ((choices[2] - 1) - choices[1]) * 10 + 32
        print('Pick : ', choices)
        print('d1 : ', d1)
        print('d2 : ', d2)
        est_tvec = scale_estimation(q1, q2, q3, d1, d2, mtx, dist)

def image_test5():
    frame = cv2.imread('./needle_detect_Img/2022-07-18_11-35-08.jpg')
    mtx, dist = camera_para_retrieve()
    tvec = aruco(frame, mtx, dist)

    for i in range(12):
        choices = [i, i+3, i+5]
        coords = np.load('coordinate.npy')
        q1 = coords[choices[0]]
        q2 = coords[choices[1]]
        q3 = coords[choices[2]]
        if choices[2] != 16:
            d1 = (choices[1] - choices[0]) * 10
            d2 = (choices[2] - choices[1]) * 10
        else:
            d1 = (choices[1] - choices[0]) * 10
            d2 = ((choices[2] - 1) - choices[1]) * 10 + 32
        print('Pick : ', choices)
        print('d1 : ', d1, 'd2 : ', d2)
        est_tvec = scale_estimation(q1, q2, q3, d1, d2, mtx, dist)

if __name__ == '__main__':
    image_test5()