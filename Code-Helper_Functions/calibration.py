# import tkinter as tk
import cv2
# from PIL import Image, ImageTk
import os
import datetime
import threading
# import imutils
# from imutils.video import VideoStream
import numpy as np
import glob
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl


# class PhotoBoothApp:
#     def __init__(self, cap, outputPath):
#         # store the video stream object and output path, then initialize
#         # the most recently read frame, thread for reading frames, and
#         # the thread stop event
#         # self.vs = vs
#         self.cap = cap
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#         self.outputPath = outputPath
#         self.frame = None
#         self.thread = None
#         self.stopEvent = None
#         # initialize the root window and image panel
#         self.root = tk.Tk()
#         self.panel = None
#         btn = tk.Button(self.root, text="Snapshot!",
#                         command=self.takeSnapshot)
#         btn.pack(side="bottom", fill="both", expand="yes", padx=10,
#                  pady=10)
#         # start a thread that constantly pools the video sensor for
#         # the most recently read frame
#         self.stopEvent = threading.Event()
#         self.thread = threading.Thread(target=self.videoLoop, args=())
#         self.thread.start()
#         # set a callback to handle when the window is closed
#         self.root.wm_title("PyImageSearch PhotoBooth")
#         self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
#
#     def videoLoop(self):
#         # DISCLAIMER:
#         # I'm not a GUI developer, nor do I even pretend to be. This
#         # try/except statement is a pretty ugly hack to get around
#         # a RunTime error that Tkinter throws due to threading
#         try:
#             # keep looping over frames until we are instructed to stop
#             while not self.stopEvent.is_set():
#                 # grab the frame from the video stream and resize it to
#                 # have a maximum width of 300 pixels
#                 ret, frame = self.cap.read()
#
#                 self.frame = cv2.flip(frame, 0)
#
#                 # self.frame = self.vs.read()
#                 # self.frame = imutils.resize(self.frame, width=300)
#
#                 # OpenCV represents images in BGR order; however PIL
#                 # represents images in RGB order, so we need to swap
#                 # the channels, then convert to PIL and ImageTk format
#                 image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
#                 image = Image.fromarray(image)
#                 image = ImageTk.PhotoImage(image)
#
#                 # if the panel is not None, we need to initialize it
#                 if self.panel is None:
#                     self.panel = tk.Label(image=image)
#                     self.panel.image = image
#                     self.panel.pack(side="left", padx=10, pady=10)
#
#                 # otherwise, simply update the panel
#                 else:
#                     self.panel.configure(image=image)
#                     self.panel.image = image
#         except RuntimeError:
#             print("[INFO] caught a RuntimeError")
#
#     def takeSnapshot(self):
#         # grab the current timestamp and use it to construct the
#         # output path
#         ts = datetime.datetime.now()
#         filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
#         p = os.path.sep.join((self.outputPath, filename))
#         # save the file
#         cv2.imwrite(p, self.frame.copy())
#         print("[INFO] saved {}".format(filename))
#
#     def onClose(self):
#         # set the stop event, cleanup the camera, and allow the rest of
#         # the quit process to continue
#         print("[INFO] closing...")
#         self.stopEvent.set()
#         self.root.quit()



def read_chessboards(images):
    """
    Charuco base pose estimation.
    """
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
    board = aruco.CharucoBoard_create(9, 5, 3, 2.4, aruco_dict)

    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners) > 0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize=(3, 3),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 1 == 0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        decimator += 1

    imsize = gray.shape
    return allCorners, allIds, imsize


def calibrate_camera(allCorners, allIds, imsize):
    """
    Calibrates the camera using the dected corners.
    """
    print("CAMERA CALIBRATION")
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
    board = aruco.CharucoBoard_create(9, 5, 3, 2.3, aruco_dict)

    cameraMatrixInit = np.array([[1000., 0., imsize[0] / 2.],
                                 [0., 1000., imsize[1] / 2.],
                                 [0., 0., 1.]])

    distCoeffsInit = np.zeros((5, 1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    # flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners=allCorners,
        charucoIds=allIds,
        board=board,
        imageSize=imsize,
        cameraMatrix=cameraMatrixInit,
        distCoeffs=distCoeffsInit,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
    print('mtx', camera_matrix)
    print('dist', distortion_coefficients0)
    print('rvec', rotation_vectors)
    print('tvec', translation_vectors)

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors


# def photosnapshot():
#     cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
#     calib_img_path = '../All_Images/calibration'
#     pba = PhotoBoothApp(cap, calib_img_path)
#     pba.root.mainloop()
#

def calibration():
    images = glob.glob('../All_images/calibration/*.jpg')
    allCorners, allIds, imsize = read_chessboards(images)
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners,allIds,imsize)
    np.save('../CameraParameter/AUX273_mtx1202.npy', mtx)
    np.save('../CameraParameter/AUX273_dist1202.npy', dist)

if __name__ == "__main__":
    calibration()
    # photosnapshot()