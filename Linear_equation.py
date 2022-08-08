import numpy as np
import math
from scipy.linalg import null_space
# import cv2.aruco as aruco
import cv2
def scale_estimation(q1, q2, q3, d1, d2):

    mtx = np.array([[2433.68459290071, 0, 718.87372815422], [0, 2440.24567073231, 601.529413632018], [0, 0, 1]])
    x_scale = (3.45 / 1000)
    y_scale = (3.45 / 1000)
    xp = mtx[0][2] * x_scale
    yp = mtx[1][2] * y_scale
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

    v1 = -(q1 - F) / np.linalg.norm((q1 - F), axis=0)
    v2 = -(q2 - F) / np.linalg.norm((q2 - F), axis=0)
    v3 = -(q3 - F) / np.linalg.norm((q3 - F), axis=0)

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
    le = p1 + unit * 22.8
    print(le)

    return le

def aruco():
    mtx = np.array([[2433.68459290071, 0, 718.87372815422], [0, 2440.24567073231, 601.529413632018], [0, 0, 1]])
    dist = np.array([[-0.0816183865808583], [-0.0605561177723738], [0.0019214883496733], [-0.00142703885621291], [1.20797639940181]])
    frame = cv2.imread('./linearImage/3.bmp')

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    squareLength = 1.67
    markerLength = 0.9
    arucoParams = cv2.aruco.DetectorParameters_create()

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=arucoParams)
    if np.any(ids != None):
        for i in range(len(ids)):

            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], markerLength, mtx, dist)

            frame = cv2.aruco.drawAxis(frame, mtx, dist, rvec, tvec, 1)  # Draw axis

            print(ids[i], rvec, tvec)

    # cv2.imshow('frame', frame)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return tvec


def undistorted():
    mtx = np.array([[2433.68459290071, 0, 718.87372815422], [0, 2440.24567073231, 601.529413632018], [0, 0, 1]])
    dist = np.array([[-0.0816183865808583], [-0.0605561177723738], [0.0019214883496733], [-0.00142703885621291], [1.20797639940181]])
    img = cv2.imread('./linearImage/3.bmp')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    cv2.imwrite('./undist3.jpg', undst)

if __name__ == '__main__':
    # undistorted()
    tvec = aruco()


    d1, d2 = 32, 40

    q1 = np.array([969, 748, 0], dtype='float64')
    q2 = np.array([904, 736, 0], dtype='float64')
    q3 = np.array([822, 722, 0], dtype='float64')
    # #


    est_tvec = scale_estimation(q1, q2, q3, d1, d2)


    err = abs((tvec - est_tvec))
    print(err)