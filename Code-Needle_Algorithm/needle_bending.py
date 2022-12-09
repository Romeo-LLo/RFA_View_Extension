
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from Linear_equation import *
import TIS
from needle_utils import *
import torch
from scipy import odr
from edge_refinement import *
import sys
sys.path.insert(0, '/home/user/Desktop/PycharmProjects/RFA_View_Extension/RFA_View_Extension/Code-Needle_Algorithm')


cfg = get_cfg()
cfg.MODEL.DEVICE = "cuda"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = '../Model_path/model_1201.pth'
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
    indicator = 0
    while True:
        if Tis.Snap_image(1) is True:
            frame = Tis.Get_image()
            frame = frame[:, :, :3]
            dis_frame = np.array(frame)
            frame = undistort_img(dis_frame, mtx, dist)

            outputs = predictor(frame)
            kp_tensor = outputs["instances"].pred_keypoints
            if kp_tensor.size(dim=0) != 0 and not torch.isnan(kp_tensor).any():


                kp = outputs["instances"].pred_keypoints.to("cpu").numpy()  # x, y, score
                x = kp[0, :-1, 0]
                y = kp[0, :-1, 1]


                if isMonotonic(x) and isMonotonic(y):
                    for i in range(11):
                        cv2.circle(frame, (int(kp[0][i][0]), int(kp[0][i][1])), 2, (0, 255, 0), -1)
                        cv2.putText(frame, str(i), (int(kp[0][i][0]), int(kp[0][i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                    (0, 0, 255), 1, cv2.LINE_AA)

                    m_, b_ = np.polyfit(x, y, 1)

                    odr_model = odr.Model(target_function)

                    data = odr.Data(x, y)
                    ordinal_distance_reg = odr.ODR(data, odr_model,
                                                   beta0=[m_, b_])
                    out = ordinal_distance_reg.run()
                    indicator = out.sum_square

                    cv2.putText(frame, str(indicator), (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.putText(frame, str(indicator), (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('Window', frame)

            if cv2.waitKey(1) == ord('q'):
                break
    Tis.Stop_pipeline()
    cv2.destroyAllWindows()


def target_function(p, x):
    m, b = p
    return m * x + b

def bending_mod():

    images = glob.glob(f'../All_images/Testimg_undist/*.jpg')

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
                coord_3D_rf, coord_2D_rf = edge_refinement_linear_mod(gray_frame, x, y, list(range(1, 9)))

                interval = []
                for i in range(1, 8):
                    dis = np.linalg.norm(coord_2D_rf[i] - coord_2D_rf[i-1])
                    interval.append(dis)
                interval[1] /= 3
                interval[4] /= 3

                std = np.std(interval)
                print(img, std)
                for coord in coord_2D_rf:
                    cv2.circle(frame, (coord[0], coord[1]), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(std), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow('point', frame)
                cv2.waitKey()
                stdList.append(std)

    print(sum(stdList) / len(stdList))
    print(max(stdList), min(stdList))


if __name__ == "__main__":
    # bending_detect()
    bending_mod()