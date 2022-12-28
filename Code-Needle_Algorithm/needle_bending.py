
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from Linear_equation import *
import TIS
from needle_utils import *
import torch
# from scipy import odr
from scipy.odr import *
from partial_filter import *
from edge_refinement import *
from collections import deque
from statistics import mean, stdev
import sys
sys.path.insert(0, '/home/user/Desktop/PycharmProjects/RFA_View_Extension/RFA_View_Extension/Code-Needle_Algorithm')


cfg = get_cfg()
cfg.MODEL.DEVICE = "cuda"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = '../Model_path/model_1221.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.TEST.DETECTIONS_PER_IMAGE = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 12
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((12, 1), dtype=float).tolist()
predictor = DefaultPredictor(cfg)
mtx, dist = camera_para_retrieve()

lazer = 1.036

def bending_detect():
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()


    loss_win = deque(maxlen=20)
    std_win = deque(maxlen=20)

    while True:
        if Tis.Snap_image(1) is True:
            frame = Tis.Get_image()
            frame = frame[:, :, :3]
            dis_frame = np.array(frame)
            frame = undistort_img(dis_frame, mtx, dist)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            outputs = predictor(frame)
            kp_tensor = outputs["instances"].pred_keypoints
            if kp_tensor.size(dim=0) != 0 and not torch.isnan(kp_tensor).any():


                kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
                x = kp[0, :-1, 0]
                y = kp[0, :-1, 1]

                odr_end_pt = 9
                for i in range(1, odr_end_pt):
                    cv2.circle(frame, (int(kp[0][i][0]), int(kp[0][i][1])), 1, (0, 0, 255), -1)
                    cv2.putText(frame, str(i), (round(kp[0][i][0]), round(kp[0][i][1])), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 0, 255), 1, cv2.LINE_AA)

                if isMonotonic(x) and isMonotonic(y) and isDistinct(x, y):

                    coord_3D_rf, coord_2D_rf = edge_refinement_linear_mod2(gray_frame, x, y, list(range(9)))

                    print(coord_2D_rf)
                    if coord_3D_rf and coord_2D_rf:
                        interval = []
                        for i in range(4, 8):
                            dis = np.linalg.norm(coord_2D_rf[i] - coord_2D_rf[i - 1])
                            interval.append(dis)
                        interval[2] /= 3
                        dis_std = np.std(interval)

                        std_win.append(dis_std)

                        odr_model = odr.Model(target_function)
                        data = odr.Data(x[1:odr_end_pt], y[1:odr_end_pt])
                        ordinal_distance_reg = odr.ODR(data, odr_model, beta0=[0., 1.])
                        out = ordinal_distance_reg.run()
                        loss = out.sum_square

                        print(loss)
                        # linear_model = Model(target_function)
                        # data = RealData(ref_x[1:], ref_y[1:])
                        #
                        # # data = RealData(x[1:odr_end_pt], y[1:odr_end_pt])
                        # odr = ODR(data, linear_model, beta0=[0., 1.])
                        # out = odr.run()
                        # m, b = out.beta
                        # print(out.sum_square, out.res_var)
                        #
                        # p1 = (0, round(b))
                        # p2 = (1000, round(m * 1000 + b))
                        # cv2.line(frame, p1, p2, (0, 0, 255), 1, cv2.LINE_AA)


                        # loss_win.append(loss)
                        # print(f"{mean(std_win):.2f}, {mean(loss_win):.2f}")
                        # print(f"{dis_std:.2f}, {loss:.2f}")
                        #
                        # if loss > 10 or loss < 0.3:
                        #     outputPath = '../All_images/bending_test/'
                        #     ts = datetime.datetime.now()
                        #     filename = "{}-{:.2f}.jpg".format(ts.strftime("%M-%S"), loss)
                        #     path = os.path.sep.join((outputPath, filename))
                        #
                        #     end_pts = end_pts_tip2end(x[1:odr_end_pt], y[1:odr_end_pt], gray_frame.shape)
                        #
                        #     cv2.line(frame, tuple(end_pts[0]), tuple(end_pts[1]), (0, 255, 0), 1, cv2.LINE_AA)
                        #     cv2.imwrite(path, frame)
                        #     print('Record')

                        for i in range(1, len(coord_2D_rf)):
                            coord = coord_2D_rf[i]
                            cv2.circle(frame, (coord[0], coord[1]), 1, (0, 255, 0), -1)

                        # cv2.putText(frame, f'dist_std : {mean(std_win):.2f}', (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # cv2.putText(frame, f'loss : {mean(loss_win):.2f}', (800, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # cv2.putText(frame, f'dist_std : {dis_std:.2f}', (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        #             (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, f'loss : {loss:.2f}', (800, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255),
                            1, cv2.LINE_AA)
            frameS = cv2.resize(frame, (1080, 810))
            cv2.imshow('Window', frameS)
            k = cv2.waitKey(30) & 0xFF
            if k == 13:
                loss_win = deque(maxlen=20)
                std_win = deque(maxlen=20)
                print('Reset')
    Tis.Stop_pipeline()
    cv2.destroyAllWindows()




def bending_detect_mod():
    Tis = TIS.TIS()
    Tis.openDevice("23224102", 1440, 1080, "30/1", TIS.SinkFormats.BGRA, True)
    Tis.Start_pipeline()


    loss_win = deque(maxlen=50)
    std_win = deque(maxlen=50)
    record = False

    while True:
        if Tis.Snap_image(1) is True:
            frame = Tis.Get_image()
            frame = frame[:, :, :3]
            dis_frame = np.array(frame)
            frame = undistort_img(dis_frame, mtx, dist)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            outputs = predictor(frame)
            kp_tensor = outputs["instances"].pred_keypoints
            if kp_tensor.size(dim=0) != 0 and not torch.isnan(kp_tensor).any():
                kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
                x = kp[0, :-1, 0]
                y = kp[0, :-1, 1]

                for i in range(11):
                    cv2.circle(frame, (int(kp[0][i][0]), int(kp[0][i][1])), 1, (0, 0, 255), -1)
                    cv2.putText(frame, str(i), (round(kp[0][i][0]), round(kp[0][i][1])), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 0, 255), 1, cv2.LINE_AA)

                m, b = line_polyfit(x, y)
                dx = x[1] - x[4]
                dy = y[1] - y[4]
                odr_end_pt = 9
                coord_2D_rf = []

                if isMonotonic(x) and isMonotonic(y) and isDistinct(x, y):
                    # for i in range(1, odr_end_pt):
                    #     cv2.circle(frame, (int(kp[0][i][0]), int(kp[0][i][1])), 1, (0, 0, 255), -1)
                    #     cv2.putText(frame, str(i), (round(kp[0][i][0]), round(kp[0][i][1])), cv2.FONT_HERSHEY_SIMPLEX,
                    #                 1.5, (0, 0, 255), 1, cv2.LINE_AA)
                    #     kernel = kernel_choice(m, i, dx, dy)
                    #     print(i, kernel)
                    #     rf_x, rf_y = edge_refinement_conv(gray_frame, x[i], y[i], kernel)
                    #     coord_2D_rf.append(np.array([rf_x, rf_y]))
                    #     cv2.circle(frame, (rf_x, rf_y), 1, (0, 255, 0), -1)

                    if coord_2D_rf:
                        interval = []
                        for i in range(1, 8):
                            dis = np.linalg.norm(coord_2D_rf[i] - coord_2D_rf[i - 1])
                            interval.append(dis)
                        interval[1] /= 3
                        interval[4] /= 3

                        dis_std = np.std(interval)

                        std_win.append(dis_std)

                        odr_model = odr.Model(target_function)
                        data = odr.Data(x[1:odr_end_pt], y[1:odr_end_pt])
                        ordinal_distance_reg = odr.ODR(data, odr_model, beta0=[0., 1.])
                        out = ordinal_distance_reg.run()
                        loss = out.sum_square

                        loss_win.append(loss)
                        # print(f"{mean(std_win):.2f}, {mean(loss_win):.2f}")
                        print(f"{dis_std:.2f}, {loss:.2f}")

                        # cv2.putText(frame, f'dist_std : {mean(std_win):.2f}', (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        #             (0, 0, 255), 1, cv2.LINE_AA)
                        # cv2.putText(frame, f'loss : {mean(loss_win):.2f}', (800, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)

                        cv2.putText(frame, f'dist_std : {dis_std:.2f}', (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                    (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, f'loss : {loss:.2f}', (800, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255),
                            1, cv2.LINE_AA)

                        if loss > 10:
                            outputPath = '../All_images/bending_test/'
                            ts = datetime.datetime.now()
                            filename = "{}-{:.2f}.jpg".format(ts.strftime("%M-%S"), loss)
                            path = os.path.sep.join((outputPath, filename))

                            cv2.imwrite(path, frame)
                            print('Record')
                            record = False

            frameS = cv2.resize(frame, (1080, 810))
            cv2.imshow('Window', frameS)
            k = cv2.waitKey(30) & 0xFF
            if k == 13:
                loss_win = deque(maxlen=20)
                std_win = deque(maxlen=20)
                print('Reset')
            if k == 27:
                record = True

    Tis.Stop_pipeline()
    cv2.destroyAllWindows()


def target_function(p, x):
    m, b = p
    return m * x + b

def bending_mod():

    images = glob.glob(f'../All_images/Testimg_undist/*.jpg')
    odr_end_pt = 10
    stdList = []
    for img in images:
        frame = cv2.imread(img)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        outputs = predictor(frame)
        kp_tensor = outputs["instances"].pred_keypoints
        if kp_tensor.size(dim=0) != 0 or not torch.isnan(kp_tensor).all():

            kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
            x = kp[0, :-1, 0]
            y = kp[0, :-1, 1]

            # m, b = line_polyfit(x, y)
            # p1 = (0, round(b))
            # p2 = (1000, round(m * 1000 + b))
            # cv2.line(frame, p1, p2, (0, 0, 255), 1, cv2.LINE_AA)
            for i in range(11):
                cv2.circle(frame, (int(kp[0][i][0]), int(kp[0][i][1])), 1, (0, 0, 255), -1)

            if isMonotonic(x) and isMonotonic(y) and isDistinct(x, y):

                if kp_tensor.size(dim=0) != 0 or not torch.isnan(kp_tensor).all():
                    kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
                    x = kp[0, :-1, 0]
                    y = kp[0, :-1, 1]

                    seq_x, seq_y, seq = partial_filter(x, y)

                coord_3D_rf, coord_2D_rf = edge_refinement_linear_mod2(gray_frame, x, y, list(range(9)))

                interval = []
                for i in range(2, 8):
                    dis = np.linalg.norm(coord_2D_rf[i] - coord_2D_rf[i-1])
                    interval.append(dis)
                interval[1] /= 3
                interval[4] /= 3

                std = np.std(interval)
                for coord in coord_2D_rf:
                    cv2.circle(frame, (coord[0], coord[1]), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(loss), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow('point', frame)
                cv2.waitKey()
                stdList.append(std)
    #
    # print(sum(stdList) / len(stdList))
    # print(max(stdList), min(stdList))

def bending_partial():

    images = glob.glob(f'../All_images/TestImg1222/*.jpg')
    for img in images:
        frame = cv2.imread(img)
        outputs = predictor(frame)
        kp_tensor = outputs["instances"].pred_keypoints
        if kp_tensor.size(dim=0) != 0 or not torch.isnan(kp_tensor).all():

            kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
            x = kp[0, :-1, 0]
            y = kp[0, :-1, 1]

            seq_x, seq_y, seq = partial_filter(x, y)
            if seq:
                m, b = line_polyfit(seq_x, seq_y)
                p1 = (0, round(b))
                p2 = (1000, round(m * 1000 + b))
                cv2.line(frame, p1, p2, (0, 0, 255), 1, cv2.LINE_AA)

                odr_model = odr.Model(target_function)
                data = odr.Data(seq_x, seq_y)
                ordinal_distance_reg = odr.ODR(data, odr_model, beta0=[m, b])
                out = ordinal_distance_reg.run()
                loss = out.sum_square
                m_, b_ = out.beta
                manual_loss = 0
                manual_loss2 = 0

                for k in range(len(seq)):
                    cv2.circle(frame, (round(seq_x[k]), round(seq_y[k])), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(seq[k]), (round(seq_x[k]), round(seq_y[k])), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 1, cv2.LINE_AA)
                    dis = np.absolute(m_*seq_x[k] - seq_y[k] + b_) / np.sqrt(m_*m_ + 1)
                    manual_loss += dis

                    # print(dis, d)
                print(loss, manual_loss, manual_loss2)

        frameS = cv2.resize(frame, (720, 540))
        cv2.imshow('win', frameS)
        cv2.waitKey()





if __name__ == "__main__":
    # bending_detect_mod()
    # bending_detect()
    bending_partial()
    # bending_mod()
    # test()