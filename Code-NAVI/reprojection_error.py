import cv2
import numpy as np
import cv2.aruco as aruco

def aruco_detection():

    mtx = np.load('./camera_parameters.npy', allow_pickle=True)[()]['mtx']
    dist = np.load('./camera_parameters.npy', allow_pickle=True)[()]['dist']
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    arucoParams = aruco.DetectorParameters_create()

    cap = cv2.VideoCapture(0)
    objps = create_objps()

    while (True):
        ret, frame = cap.read()
        if ret == True:
            corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=arucoParams)
            aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))

            if np.any(ids != None):
                if(len(ids) == 4):
                    ids_copy = ids.flatten()
                    order = np.argsort(ids_copy)
                    corners = [corners[i] for i in order]
                    corners = np.squeeze(np.array(corners), axis=1)
                    img_pts = corners.reshape((-1, 1, 2))

                    rvec, tvec = solvePnP_nPoints(objps, img_pts, mtx, dist)
                    frame = aruco.drawAxis(frame, mtx, dist, rvec, tvec, 1)

                    detect_pts = corners.reshape(16, 2)
                    proj_pts, _ = cv2.projectPoints(objps, rvec, tvec, mtx, dist)
                    proj_pts = proj_pts.squeeze(1)
                    norm = np.linalg.norm(detect_pts - proj_pts, keepdims=True, axis=1)
                    diff = np.sum(norm, axis=0)
                    if diff > 100:
                        print("warning ", diff)
                    else:
                        print(diff)




            pressedKey = cv2.waitKey(1) & 0xFF

            if pressedKey == ord('q'):
                # excel_file = 'C:/Users/HP/Desktop/Niddle_displacement.xls'
                # workbook.save(excel_file)
                break

            cv2.imshow("Image", frame)

        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def create_objps():
    square_len = 1.3   # a
    marker_len = 0.95  # b

    k = marker_len / 2
    m = square_len + marker_len / 2
    n = square_len - marker_len / 2

    objps = np.array([[-k, m, 0],
                      [k, m, 0],
                      [k, n, 0],
                      [-k, n, 0],

                      [-m, k, 0],
                      [-n, k, 0],
                      [-n, -k, 0],
                      [-m, -k, 0],

                      [n, k, 0],
                      [m, k, 0],
                      [m, -k, 0],
                      [n, -k, 0],

                      [-k, -n, 0],
                      [k, -n, 0],
                      [k, -m, 0],
                      [-k, -m, 0]], dtype=np.float32)
    new_objps = objps.copy()

    total_offset = 0
    for i in range(4):
        scale = 0.2
        offset_x = marker_len * np.random.uniform(-scale, scale)
        offset_y = marker_len * np.random.uniform(-scale, scale)
        offset_z = marker_len * np.random.uniform(-scale, scale)

        offset = np.sqrt(offset_x ** 2 + offset_y ** 2 + offset_z ** 2)
        print(offset)
        total_offset += offset
        # offset = marker_len * 0.5

        new_objps[4 * i, 0] += offset_x
        new_objps[4 * i + 1, 0] += offset_x
        new_objps[4 * i + 2, 0] += offset_x
        new_objps[4 * i + 3, 0] += offset_x

        new_objps[4 * i, 1] += offset_y
        new_objps[4 * i + 1, 1] += offset_y
        new_objps[4 * i + 2, 1] += offset_y
        new_objps[4 * i + 3, 1] += offset_y

        new_objps[4 * i, 2] += offset_z
        new_objps[4 * i + 1, 2] += offset_z
        new_objps[4 * i + 2, 2] += offset_z
        new_objps[4 * i + 3, 2] += offset_z
    print("total offsest : ", total_offset)
    return new_objps

def solvePnP_nPoints(objps, imgps, mtx, dist):



    ret_val, rotation_vector, translation_vector = cv2.solvePnP(objps, imgps, mtx, dist)

    return rotation_vector, translation_vector

if __name__ == '__main__':
    aruco_detection()
    #total offset<0.3 以內不會有炸裂的error
    #但>0.3 有些set會炸裂，但有些還是會穩定，非常奇怪

