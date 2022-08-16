import cv2.aruco as aruco
import cv2
import glob
import os
import numpy as np
import json
from numpy import linalg as LA
from scipy.optimize import least_squares, leastsq

def least_sq():
    id_list, x0 = load_pts3D()
    res = least_squares(fun, x0, method='lm')
    print(res.x)

def fun(x):
    path = "./Transducer"
    imgs = glob.glob(os.path.join(path, '*.bmp'))
    mtx, dist = load_para()

    objps_set = x.reshape(-1, 3)
    for i, img in enumerate(imgs):

        rvec, tvec, objps, imgps = corners_detection(img, objps_set)
        imgpts_proj, _ = cv2.projectPoints(objps, rvec, tvec, mtx, dist)
        error_each_img = LA.norm(imgpts_proj.squeeze(axis=1) - imgps, axis=1)
        if i == 0:
            error_whole_pts = error_each_img
        else:
            error_whole_pts = np.concatenate((error_whole_pts, error_each_img))

    error_sum = np.sum(error_whole_pts, axis=0)
    print(error_sum)

        # reproj_error = np.sum(LA.norm(imgpts_proj.squeeze(axis=1) - imgps, axis=1), axis=0) # sum error works bad
        # total_error += reproj_error

    return error_whole_pts


def fun2(x):
    path = "./Transducer"
    imgs = glob.glob(os.path.join(path, '*.bmp'))
    mtx, dist = load_para()

    objps_set = x.reshape(-1, 3)
    for i, img in enumerate(imgs):

        rvec, tvec, objps, imgps = corners_detection(img, objps_set)
        imgpts_proj, _ = cv2.projectPoints(objps, rvec, tvec, mtx, dist)
        error_each_img = LA.norm(imgpts_proj.squeeze(axis=1) - imgps, axis=1)
        if i == 0:
            error_whole_pts = error_each_img
        else:
            error_whole_pts = np.concatenate((error_whole_pts, error_each_img))

    error_sum = np.sum(error_whole_pts, axis=0)
    print(error_sum)

        # reproj_error = np.sum(LA.norm(imgpts_proj.squeeze(axis=1) - imgps, axis=1), axis=0) # sum error works bad
        # total_error += reproj_error

    return error_whole_pts

def corners_detection(img_path, objps_set):


    mtx, dist = load_para()
    img_array = cv2.imread(img_path)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    arucoParams = cv2.aruco.DetectorParameters_create()

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img_array, aruco_dict, parameters=arucoParams)
    imgps = np.zeros((len(ids) * 4, 2))
    objps = np.zeros((len(ids) * 4, 3))

    id_list = np.load('./id_list.npy')

    for i in range(len(ids)):
        index = np.where(id_list == str(ids[i][0]))[0]
        for j in range(4):
            imgps[4 * i + j] = np.array((corners[i][0][j]))
            objps[4 * i + j] = objps_set[4 * index + j]
    ret, rvec, tvec = cv2.solvePnP(objps, imgps, mtx, dist)

    return rvec, tvec, objps, imgps




def load_para():
    path = "./Transducer"
    file = os.path.join(path, 'camera parameters.txt')
    # ins_paremeter = [json.loads(line) for line in open(file, 'r')]

    with open(file) as f:
        ins_paremeter = json.load(f)
    mtx = np.array([[ins_paremeter['FX'], 0, ins_paremeter['CX']], [0, ins_paremeter['FY'], ins_paremeter['CY']], [0, 0, 1]])
    dist = np.array([[ins_paremeter['D0']], [ins_paremeter['D1']], [ins_paremeter['D2']],  [ins_paremeter['D3']],  [ins_paremeter['D4']]])


    return mtx, dist

def load_constrained_pts3D():
    path = './Transducer/parameter.json'
    with open(path) as f:
        data = json.load(f)
        pts3D = data["TrackerRelatedPositions"]['Transducer']

        id_list = list(data["TrackerRelatedPositions"]['Transducer'].keys())
        objps_center_set = np.zeros((len(id_list), 3))
        objps_vector_set = np.zeros((len(id_list), 3))
        objps_norvec_set = np.zeros((len(id_list), 3))

        for i, id in enumerate(id_list):
            center_x = (pts3D[id][0]['X'] + pts3D[id][1]['X'] + pts3D[id][2]['X'] + pts3D[id][3]['X']) / 4
            center_y = (pts3D[id][0]['Y'] + pts3D[id][1]['Y'] + pts3D[id][2]['Y'] + pts3D[id][3]['Y']) / 4
            center_z = (pts3D[id][0]['Z'] + pts3D[id][1]['Z'] + pts3D[id][2]['Z'] + pts3D[id][3]['Z']) / 4

            vector_x = pts3D[id][0]['X'] - center_x
            vector_y = pts3D[id][0]['Y'] - center_y
            vector_z = pts3D[id][0]['Z'] - center_z

            vector_x_2 = pts3D[id][1]['X'] - center_x
            vector_y_2 = pts3D[id][1]['Y'] - center_y
            vector_z_2 = pts3D[id][1]['Z'] - center_z

            norvec = np.cross(np.array((vector_x_2, vector_y_2, vector_z_2)), np.array((vector_x, vector_y, vector_z)))
            #出紙面方向

            objps_center_set[i] = np.array((center_x, center_y, center_z))
            objps_vector_set[i] = np.array((vector_x, vector_y, vector_z))
            objps_norvec_set[i] = norvec / np.linalg.norm(norvec)


        return id_list, objps_center_set, objps_vector_set, objps_norvec_set

def recover_pts3D(id_list, objps_center_set, objps_vector_set, objps_norvec_set):
    objps_set = np.zeros((len(id_list) * 4, 3))
    for i in range(len(id_list)):
            vector_0 = objps_vector_set[i]
            vector_1 = np.cross(vector_0, objps_norvec_set[i])
            vector_2 = np.cross(vector_1, objps_norvec_set[i])
            vector_3 = np.cross(vector_2, objps_norvec_set[i])

            objps_set[4 * i] = objps_center_set[i] + vector_0
            objps_set[4 * i + 1] = objps_center_set[i] + vector_1
            objps_set[4 * i + 2] = objps_center_set[i] + vector_2
            objps_set[4 * i + 3] = objps_center_set[i] + vector_3
    return objps_set

def load_pts3D():
    path = './Transducer/parameter.json'
    with open(path) as f:
        data = json.load(f)
        pts3D = data["TrackerRelatedPositions"]['Transducer']

        id_list = list(data["TrackerRelatedPositions"]['Transducer'].keys())
        objps_set = np.zeros((len(id_list) * 4, 3))

        for i, id in enumerate(id_list):
            for j in range(4):
                objps_set[4 * i + j] = np.array((pts3D[id][j]['X'], pts3D[id][j]['Y'], pts3D[id][j]['Z']))
        objps_set = objps_set.flatten()
    return id_list, objps_set

def save_id_list(id_list):
    np.save('./id_list.npy', np.array(id_list))


if __name__ == "__main__":
    load_constrained_pts3D()
    # id_list, objps_set = load_pts3D()
    # least_sq()
#


#TODO
# objps要改成 [N, 3] array 形式
