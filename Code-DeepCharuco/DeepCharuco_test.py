import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import glob
import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torchvision.models as models
import pandas as pd
import cv2
import random
from DeepCharuco_Model import DeepCharuco
import matplotlib
import time
from math import atan2
import cv2.aruco as aruco

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.csv = pd.read_csv('TempImage/output.csv')
        self.sample_num = self.csv.shape[0]       # 以csv的數量為主，照片可能會比較多
        files = glob.glob(os.path.join(root, '*.jpg'))
        files.sort()
        self.files = files[:self.sample_num]
        self.len = len(self.files)

        img_fn = self.files[0]
        img = Image.open(img_fn)
        self.cell_size = 8
        self.width = img.size[0]
        self.height = img.size[1]
        self.x_cells = int(self.width / self.cell_size)
        self.y_cells = int(self.height  / self.cell_size)

    def __getitem__(self, index):
        img_fn = self.files[index]
        img = Image.open(img_fn)
        coords = self.csv.iloc[index, 1:]
        label2D = self.coord2binary(coords)
        id2D = self.idto2D(coords)

        if self.transform is not None:
            img = self.transform(img)

        return img, label2D, id2D

    def coord2binary(self, coords):
        label2D = torch.zeros(self.height, self.width)  # 480*640
        for i in range(4):
            y = round(coords[2*i+1])
            x = round(coords[2*i])
            label2D[y, x] = 1
        return label2D

    def idto2D(self, coords):
        id2D = torch.zeros(self.y_cells, self.x_cells)  # 0 stands for no id
        for i in range(4):
            x = round(coords[2*i] // self.cell_size)
            y = round(coords[2*i+1] // self.cell_size)
            id2D[y, x] = i + 1
        return id2D


    def __len__(self):
        return self.len

def idto2D(coords):
    cell_size = 8
    y_cells, x_cells = 60, 80
    id2D = torch.zeros(y_cells, x_cells)  # 0 stands for no id
    for i in range(4):
        x = round(coords[2*i] // cell_size)
        y = round(coords[2*i+1] // cell_size)
        id2D[y, x] = i + 1
    return id2D

def imshow(img):
    img = img.numpy()
    print(img.shape)
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

def imgValid(img, id2D):
    id = id2D[0]
    showimg = img.numpy()
    for h in range(60):
        for w in range(80):
            index = id[h, w]   # 0 is the first item of the batch

            if index != 0:
                y = h * 8 + 4
                x = w * 8 + 4
                print(index, y, x)
                plt.text(x, y, str(int(index.item())), fontsize=5, bbox=dict(facecolor="r"))

    plt.imshow(np.transpose(showimg.squeeze(0), (1, 2, 0)), cmap='gray')
    plt.show()

def labels2Dto3D_flattened(labels, cell_size):

    batch_size, channel, H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(8)
    labels = space2depth(labels).cuda()
    dustbin = torch.ones((batch_size, 1, Hc, Wc)).cuda()
    # labels = torch.cat((labels*2, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)  # why times 2
    labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)

    labels = torch.argmax(labels, dim=1)
    return labels

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output

def SetupTrain():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepCharuco()
    model = model.to(device)
    return model




def test_show(model_dir, test_dir):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepCharuco()
    model = model.to(device)
    checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))

    size = 1.3
    mtx = np.load('./camera_parameters.npy', allow_pickle=True)[()]['mtx']
    dist = np.load('./camera_parameters.npy', allow_pickle=True)[()]['dist']

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    for number in range(1, 10):
        start = time.time()
        filename = "./TempImage/{}.jpg".format(number)

        # filename = "./TempImage/1.jpg"

        # coords = csv.iloc[number, 1:]
        # target_id = idto2D(coords).unsqueeze(0).to(device)

        # img = Image.open(filename).convert('L')
        for k in range(1):
            if k == 0:
                print("Distorted")
                img_cv = cv2.imread(filename, 0)
            else:
                print("Undistorted")
                img_cv = cv2.imread(filename, 0)
                img_cv = undistort_img(img_cv, mtx, dist)

            transform = transforms.ToTensor()
            img = transform(img_cv).unsqueeze(0).to(device) # this can be count!?

            out_loc = model(img)['semi']
            # out_id = model(img)['desc']

            step_net = time.time()
            # criterion = nn.CrossEntropyLoss()
            # id_loss = criterion(out_id, target_id.type(torch.int64))

            # print("id_loss = {:.4f}".format(id_loss))
            pred_loc = torch.argmax(out_loc, dim=1).cpu()
            # pred_id = torch.max(out_id, dim=1)[1].cpu().numpy()

            x, y = restore_coord(pred_loc)
            # id = [pred_id[0, y, x] for y, x in zip(cell_loc_y, cell_loc_x)]

            x_inorder, y_inorder = order_points_clockwise(x, y)
            new_x, new_y = correct_order_by_white_v2(img_cv, x_inorder, y_inorder)


            image_points_rf = refinement(img_cv, new_x, new_y)
            rotation_vector_rf, translation_vector_rf = solvePnP(size, image_points_rf, mtx, dist)

            # new_coord = np.concatenate((new_x, new_y), axis=0)
            # new_coord = new_coord.reshape(2, 4)
            # image_points = new_coord.transpose()
            # image_points = image_points.astype('float32')

            ## draw circle on the found corners

            # showimg = Image.open(filename)
            # showimg = transform(showimg).unsqueeze(0)
            # showimg = showimg.numpy()

            # fig, ax = plt.subplots(1)
            # ax.set_aspect('equal')
            # ax.imshow(np.transpose(showimg.squeeze(0), (1, 2, 0)), cmap='gray')
            # cms = matplotlib.cm
            #
            # for i in range(4):
            #         #     x_, y_ = new_x[i], new_y[i]
            #         #
            #         #     circ = plt.Circle((x_, y_), 2*i + 1, fill=True, color=cms.jet(0.9))
            #         #     ax.add_patch(circ)
            # plt.show()



            new_coord = np.concatenate((new_x, new_y), axis=0)
            new_coord = new_coord.reshape(2, 4)
            image_points = new_coord.transpose()
            image_points = image_points.astype('float32')
            rotation_vector, translation_vector = solvePnP(size, image_points, mtx, dist)

            end = time.time()
            net_time = step_net - start
            total_time = end - start
            fps = 1 / total_time
            # print(fps, " ", net_time / total_time)

            ## aruco detection
            diamondCorners, rvec, tvec = diamond_detection(img_cv, mtx, dist)
            if diamondCorners == None:
                print(f'Number{number}, can not detect')
                print("_____________________")
            else:
                print(f'Number{number}, detected')
                pix_error, dis_error = error_calc(diamondCorners, image_points, rvec, tvec, rotation_vector, translation_vector)
                pix_error_rf, dis_error_rf = error_calc(diamondCorners, image_points_rf, rvec, tvec, rotation_vector_rf, translation_vector_rf)
                pix_improv = (pix_error - pix_error_rf) / pix_error
                dis_improv = (dis_error - dis_error_rf) / dis_error
                print(f'Pix improv {pix_improv  }, Dis improv {dis_improv}')
                print("_____________________")





            del out_loc
            del img, img_cv

def refinement(img, x, y):

    image_points = np.zeros((1, 4, 2), dtype="float32")
    for i in range(4):
        image_points[0, i, 0] = x[i]
        image_points[0, i, 1] = y[i]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(img, image_points, (5, 5), (-1, -1), criteria)
    return corners[0]

def restore_coord(pred_loc):
    _, cell_loc_y, cell_loc_x = (pred_loc != 64).nonzero(as_tuple=True)
    spec_y = [pred_loc[0, y, x] // 8 for y, x in zip(cell_loc_y, cell_loc_x)]
    spec_x = [pred_loc[0, y, x] % 8 for y, x in zip(cell_loc_y, cell_loc_x)]
    loc_y = cell_loc_y * 8
    loc_x = cell_loc_x * 8

    x = loc_x.numpy() + np.array(spec_x)
    y = loc_y.numpy() + np.array(spec_y)

    return x, y
def order_points_clockwise(x, y):
    coord = np.concatenate((x, y), axis=0)
    coord = coord.reshape(2, 4)
    coord_sum = np.sum(coord, axis=0)
    tl = np.argmin(coord_sum)
    br = np.argmax(coord_sum)
    coord_list = [0, 1, 2, 3]
    coord_list.remove(tl)
    coord_list.remove(br)

    tr = coord_list[0] if (x[coord_list[0]] - x[tl]) > (x[coord_list[1]] - x[tl]) else coord_list[1]
    coord_list.remove(tr)
    bl = coord_list[0]
    order = [tl, tr, br, bl]
    # clock-wise order, once we found the white

    # This is the method for drawing
    x_inorder = [x[ord] for ord in order]
    y_inorder = [y[ord] for ord in order]

    return x_inorder, y_inorder

def correct_order_by_white_v2(img, x, y):
    x_center_main = (int) (np.mean(x))
    y_center_main = (int) (np.mean(y))
    p_c = [x_center_main, y_center_main]
    sum_list = []
    for i in range(4):
        p = [x[i], y[i]]
        pixel_sum = pixelsum_between_2p(img, p, p_c)
        sum_list.append(pixel_sum)
    white_id = sum_list.index(max(sum_list))

    new_x = rotate(x, white_id)
    new_y = rotate(y, white_id)

    return new_x, new_y




def pixelsum_between_2p(img, p, q):
    delta_x = p[0] - q[0]
    delta_y = p[1] - q[1]
    m = delta_y / delta_x
    b = p[1] - m * p[0]
    pixel_sum = 0
    if delta_x > 0:
        for x in range(q[0], p[0]):
            y = int(m * x + b)
            pixel_sum += img[y, x]
    else:
        for x in range(p[0], q[0]):
            y = int(m * x + b)
            pixel_sum += img[y, x]

    return pixel_sum



def rotate(l, n):
    return l[n:] + l[:n]

def solvePnP(size, image_points, mtx, dist):
    objps = np.array([
            [-size / 2, size / 2, 0],
            [size / 2, size / 2, 0],
            [size / 2, -size / 2, 0],
            [-size / 2, -size / 2, 0]], dtype=np.float32)



    ret_val, rotation_vector, translation_vector = cv2.solvePnP(objps, image_points, mtx, dist)

    return rotation_vector, translation_vector

def error_calc(diamondCorners, image_points, rvec, tvec, rotation_vector, translation_vector):
    diamondCorners = diamondCorners[0].squeeze(1)
    pix_error = 0
    for i, corner in enumerate(diamondCorners):
        pix_error += np.linalg.norm(corner - image_points[i])

    a = tvec[0][0] - translation_vector.T[0]
    print("gt : ",  tvec[0][0])
    print("est : ",  translation_vector.T[0])

    dis_error = np.linalg.norm(a)

    print(f"pix_error = {pix_error}, dis_error = {dis_error}")
    return pix_error, dis_error


def undistort_img(img, mtx, dist):
    h1, w1 = img.shape[:2]

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst

def diamond_detection(img, mtx, dist):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    squareLength = 1.3
    markerLength = 0.9
    arucoParams = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=arucoParams)

    if np.any(ids != None):
        diamondCorners, diamondIds = aruco.detectCharucoDiamond(img, corners, ids,
                                                                squareLength / markerLength)

        if np.any(diamondIds != None):  # if aruco marker detected
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(diamondCorners, squareLength, mtx, dist)  # For a single marker
            # rvec, tvec = solvePnP()
            return diamondCorners, rvec, tvec
        else:
            return None, None, None
    else:
        return None, None, None

def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot


def test_loc_loss():
    testset = CustomDataset(root='TestImage', transform=transforms.ToTensor())
    testset_loader = DataLoader(testset, batch_size=16, shuffle=True, num_workers=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepCharuco()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999))
    checkpoint = torch.load('Model_dict/epoch40.pth', map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()


    for batch_id, (input, target_label2D, target_id) in enumerate(testset_loader):
        input, target_label2D, target_id = input.to(device), target_label2D.to(device), target_id.to(device)
        optimizer.zero_grad()
        pred_loc = model(input)['semi']
        pred_id = model(input)['desc']

        target_loc = labels2Dto3D_flattened(target_label2D.unsqueeze(1), 8)


def test_stream():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepCharuco()
    model = model.to(device)
    model_dir = 'Model_dict/epoch100.pth'
    checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    img_w, img_h = 480, 360

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    start = time.time()
    num_frame = 0

    while (True):
        ret, frame = cap.read()
        if ret:
            num_frame += 1
            imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            transform = transforms.ToTensor()
            imgGray = transform(imgGray).unsqueeze(0).to(device)

            out_loc = model(imgGray)['semi']
            pred_loc = torch.argmax(out_loc, dim=1).cpu()

            _, cell_loc_y, cell_loc_x = (pred_loc != 64).nonzero(as_tuple=True)
            # _, cell_id_y, cell_id_x = (pred_id != 0).nonzero(as_tuple=True)
            # print('x align : {}, y align : {}'.format(cell_id_x == cell_loc_x, cell_id_y == cell_loc_y))
            spec_y = [pred_loc[0, y, x] // 8 for y, x in zip(cell_loc_y, cell_loc_x)]
            spec_x = [pred_loc[0, y, x] % 8 for y, x in zip(cell_loc_y, cell_loc_x)]
            loc_y = cell_loc_y * 8
            loc_x = cell_loc_x * 8

            y = loc_y.numpy() + np.array(spec_y)
            x = loc_x.numpy() + np.array(spec_x)

            for i in range(len(x)):
                x_, y_ = x[i], y[i]
                cv2.circle(frame, (x_, y_), i+2, (255, 0, 0), -1)

            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
            squareLength = 1.67
            markerLength = 1
            arucoParams = aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=arucoParams)

            if np.any(ids != None):
                diamondCorners, diamondIds = aruco.detectCharucoDiamond(frame, corners, ids,
                                                                        squareLength / markerLength)
                print('aruco detected')
            else:
                print('No!')
            # cnt = np.array((x, y)).T
            # rect = cv2.minAreaRect(cnt)
            #
            # img_crop, img_rot = crop_rect(imgGray, rect)
            #
            # coord = np.concatenate((x, y), axis=0)
            # coord = coord.reshape(2, 4)
            # coord_sum = np.sum(coord, axis=0)
            # tl = np.argmin(coord_sum)
            # br = np.argmax(coord_sum)
            # coord_list = [0, 1, 2, 3]
            # coord_list.remove(tl)
            # coord_list.remove(br)
            #
            # tr = coord_list[0] if (x[coord_list[0]] - x[tl]) > (x[coord_list[1]] - x[tl]) else coord_list[1]
            # coord_list.remove(tr)
            # bl = coord_list[0]
            # order = [tl, tr, br, bl]
            #
            # end = time.time()
            # seconds = end - start
            # fps = num_frame / seconds

            # print('FPS : ', fps)


            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord('q'):
                break
            cv2.imshow("Image", frame)

        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_dir = 'Model_dict/epoch60.pth'
    test_dir = 'TempImage'
    test_show(model_dir, test_dir)
    # test_stream()

