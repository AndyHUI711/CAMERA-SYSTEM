import csv
import numpy as np
import time
import cv2
import cv2.aruco as aruco
import math
from scipy.spatial.distance import euclidean
from PIL import ImageEnhance, Image
import imutils
import sys
import os
import argparse
#import calculate
#import csv

#加载鱼眼镜头的yaml标定文件
#加载相机纠正参数
cv_file = cv2.FileStorage("camera.yaml", cv2.FILE_STORAGE_READ)
camera_matrix = cv_file.getNode("camera_matrix").mat()
dist_matrix = cv_file.getNode("dist_coeff").mat()
cv_file.release()
#加载默认cam参数
dist=np.array(([[-0.01337232,  0.01314211, -0.00060755, -0.00497024,  0.08519319]]))
newcameramtx=np.array([[484.55267334,   0.,         325.60812827],
[  0.,         480.50973511, 258.93040826],
[  0.,           0.,           1.        ]])
mtx=np.array( [[428.03839374,   0,         339.37509535],
[  0.,         427.81724311, 244.15085121],
[  0.,           0.,          1.        ]])

##以上数据基于相机以及其他py程序


cap = cv2.VideoCapture(1)
#cap.set(3, 1280)  # 设置分辨率
#cap.set(4, 768)


#0:computer cam/1:usb cam
font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)
CACHED_PTS = None
CACHED_IDS = None
Line_Pts = None
measure = None
index = 0
def enhance(image):
    # 亮度增强
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 0.3
    image_brightened = enh_bri.enhance(brightness)
    # image_brightened.show()

    # 色度增强
    enh_col = ImageEnhance.Color(image_brightened)
    color = 2
    image_colored = enh_col.enhance(color)
    # image_colored.show()

    # 对比度增强
    enh_con = ImageEnhance.Contrast(image_colored)
    contrast = 5
    image_contrasted = enh_con.enhance(contrast)
    # image_contrasted.show()

    # 锐度增强
    enh_sha = ImageEnhance.Sharpness(image_contrasted)
    sharpness = 3.0
    image_sharped = enh_sha.enhance(sharpness)
    # image_sharped.show()

    return image_sharped
while cap.isOpened():
    Dist = []
    ret, frame = cap.read()
    if ret:
        assert not isinstance(frame, type(None)), 'frame not found'
    # 获取视频帧的高

    h1 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 获取视频帧的宽
    w1 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(h1/2)
    w2 = int(w1 / 2)
    #print(h1,w1,h2,w2) #480 640 240 320
    cam_coordinate=(h2,w2)
    # 纠正畸变
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_matrix, (h1, w1), 0, (h1, w1))
    frame = cv2.undistort(frame, camera_matrix, dist_matrix, None, newcameramtx)

    cv2.line(frame, (w2 - 5, h2), (w2 + 5, h2), (0, 0, 225), 1)
    cv2.line(frame, (w2, h2 - 5), (w2, h2 + 5), (0, 0, 225), 1)



    #x, y, w1, h1 = roi
    #dst1 = dst1[y:y + h1, x:x + w1]

    #灰度化，检测aruco标签，所用字典为DICT_ARUCO_ORIGINAL
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    retval, gray_otsu = cv2.threshold(gray, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    gaussian1 = cv2.GaussianBlur(gray, (5, 5), 0)
    gaussian1_enhance = Image.fromarray(np.uint8(gaussian1))
    gaussian1_enhance = enhance(gaussian1_enhance)

    #retval 最合适阈值
    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    parameters =  aruco.DetectorParameters_create()

    #使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        np.asarray(gaussian1_enhance),aruco_dict,parameters=parameters)

    #print(corners)
    if len(corners) <= 0:
        if CACHED_PTS is not None:
            corners = CACHED_PTS
    if len(corners) > 0:
        CACHED_PTS = corners
        if ids is not None:
            ids = ids.flatten()
            CACHED_IDS = ids
        else:
            if CACHED_IDS is not None:
                ids = CACHED_IDS
        if len(corners) < 2:
            if len(CACHED_PTS) >= 2:
                corners = CACHED_PTS
        for (markerCorner, markerId) in zip(corners, ids):
            #print("[INFO] Marker detected")
            corners_abcd = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners_abcd
            #print(corners_abcd)
            topRightPoint = (int(topRight[0]), int(topRight[1]))
            topLeftPoint = (int(topLeft[0]), int(topLeft[1]))
            bottomRightPoint = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeftPoint = (int(bottomLeft[0]), int(bottomLeft[1]))

            cX = int((topLeft[0] + bottomRight[0]) // 2)
            cY = int((topLeft[1] + bottomRight[1]) // 2)
            measure = abs(3.5/(topLeft[0]-cX))
            cv2.circle(frame, (int(cX), int(cY)), 4, (255, 0, 0), -1)

            #cv2.putText(frame, str(
            #    int(markerId)), (int(topLeft[0]-10), int(topLeft[1]-10)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            Dist.append((cX, cY))
            # print(arucoDict)

            if len(Dist) == 0:
                if Line_Pts is not None:
                   Dist = Line_Pts
            if len(Dist) == 2:
                Line_Pts = Dist
            if len(Dist) == 2:
                cv2.line(frame, Dist[0], Dist[1], (255, 0, 255), 2)
                ed = (1/7.5)*1.8*((Dist[0][0] - Dist[1][0])**2 + ((Dist[0][1] - Dist[1][1])**2))**(0.5)
                cv2.putText(frame, str(float(measure*(ed))) + "cm", (int(300), int(
              300)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

    #headers = ['index', 'x', 'y', 'topdis', 'bottomdis', 'rightdis', 'leftdis']
    index = 0
    if ids is not None:
        # 获取aruco返回的rvec旋转矩阵、tvec位移矩阵
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_matrix)
        (rvec - tvec).any()  # get rid of that nasty numpy value array error
        # print(rvec)

        # 在画面上 标注auruco标签的各轴
        for i in range(rvec.shape[0]):
            aruco.drawAxis(frame, camera_matrix, dist_matrix, rvec[i, :, :], tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(frame, corners, ids)
            c = corners[i][0]
            cx = float(c[:, 0].mean())
            cy = float(c[:, 1].mean())
            coordinate = (cx, cy)
            cv2.circle(frame, (int(cx), int(cy)), 2, (255, 255, 0), 2)

            (topLeft, topRight, bottomRight, bottomLeft) = corners[0][0][0], corners[0][0][1], \
                                                           corners[0][0][2], \
                                                           corners[0][0][3]
            topRightx = np.array((float(topRight[0]), float(topRight[1])))
            bottomRightx = np.array((float(bottomRight[0]), float(bottomRight[1])))
            bottomLeftx = np.array((float(bottomLeft[0]), float(bottomLeft[1])))
            topLeftx = np.array((float(topLeft[0]), float(topLeft[1])))

            right = topRightx - bottomRightx
            left = topLeftx - bottomLeftx
            top = topLeftx - topRightx
            bottom = bottomLeftx - bottomRightx
            rightdis = math.hypot(right[0], right[1])
            leftdis = math.hypot(left[0], left[1])
            topdis = math.hypot(top[0], top[1])
            bottomdis = math.hypot(bottom[0], bottom[1])

            average4 = (rightdis + leftdis + topdis + bottomdis) / 4
            averagecol = (rightdis + leftdis) / 2
            averagerow = (topdis + bottomdis) / 2

            # print(rightdis, leftdis, topdis, bottomdis)
            # y,x,rightdis,leftdis,topdis,bottomdis,average4,averagecol,averagerow
            values = [index, cy, cx, topdis, bottomdis, rightdis, leftdis, average4, averagecol, averagerow]

            # with open('markers_data2.csv', 'a+', newline='') as fp:
            with open('markers_data2.csv', 'a+', newline='') as fp:
                # 获取对象
                write = csv.writer(fp)
                # print(cx,cy)
                # write.writerow(headers)
                write.writerow(values)
                index = index + 1

    else:
        ##### DRAW "NO IDS" #####
        cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
        #cv2.putText(gray_otsu, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


    # 显示结果画面

   #cv2.imshow("frame", new_frame)

    #cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    cv2.putText(frame, "Id: " + str(ids), (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    frame = cv2.resize(frame, (0, 0), fx=1, fy=1,
                             interpolation=cv2.INTER_NEAREST)
    cv2.imshow("frame",frame)
    cv2.imshow("enhance", np.asarray(gaussian1_enhance))

    key = cv2.waitKey(1)

    if key == 27:         # 按esc键退出
        print('esc break...')
        cap.release()
        cv2.destroyAllWindows()
        fp.close()
        break

    if key == ord(' '):   # 按空格键保存
#        num = num + 1
#        filename = "frames_%s.jpg" % num  # 保存一张图像
        filename = str(time.time())[:10] + ".jpg"
        cv2.imwrite(filename, frame)