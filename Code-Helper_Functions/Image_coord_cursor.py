# importing the module
import cv2
import numpy as np

# function to display the coordinates of
# of the points clicked on the image

coordinates = np.zeros((1, 3), dtype='float64')
index = 0
def click_event(event, x, y, flags, params):
    global coordinates, index
    # button = [20, 60, 50, 250]
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        if y > button[0] and y < button[1] and x > button[2] and x < button[3]:
            print('Save coordinates!')
            print(coordinates)
            np.save('coordinate.npy', coordinates)
        cv2.imshow('image', combine_img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        if index == 0:
            coordinates[0] = np.array([x, y, 0], dtype='float64')
            index += 1
        else:
            coord = np.array([[x, y, 0]], dtype='float64')
            coordinates = np.concatenate((coordinates, coord), axis=0)
        print(x, ' ', y)

        cv2.circle(combine_img, (x, y), 2, (0, 0, 255), -1)
        cv2.imshow('image', combine_img)


# driver function
if __name__ == "__main__":

    # y 768 x 1024
    img = cv2.imread('./needle_detect_Img/2022-07-18_11-35-08.jpg')
    bt_size = 150
    button = [0, bt_size, img.shape[1], img.shape[1] + bt_size]  # y, x

    control_image = np.zeros((img.shape[0], bt_size, 3), np.uint8)
    control_image[:bt_size, :bt_size, :] = 180
    cv2.putText(control_image, 'Button', (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)

    combine_img = np.concatenate((img, control_image), axis=1)

    cv2.imshow('image', combine_img)
    cv2.setMouseCallback('image', click_event)

    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()
