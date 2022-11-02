import cv2


cap = cv2.VideoCapture('../All_images/test07.mp4')
count = 0
while cap.isOpened() and count <= 10:
    ret, image = cap.read()
    cv2.imwrite("../All_images/frame%d.jpg" % count, image)  # save frame as JPEG file
    print('done')
    count += 1