import numpy as np
import cv2

video = cv2.VideoCapture("road_car_view.mp4")

while True:
     ret, frame = video.read()
     if not ret:
          video = cv2.VideoCapture("road_car_view.mp4")
          continue

     size = cv2.resize(frame, (800, 600))
     
     hsv = cv2.cvtColor(size, cv2.COLOR_BGR2HSV)
     blur = cv2.GaussianBlur(hsv, (5, 5), 0)
     
     lower_yellow = np.array([20, 84, 140])
     upper_yellow = np.array([58,245, 238])

     mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
     edges = cv2.Canny(mask, 75, 150)

     lines = cv2.HoughLinesP(edges, 1, np.pi/180, 70, maxLineGap = 50)
     if lines is not None:
          for line in lines:
               x1, y1, x2, y2 = line[0]
               cv2.line(size, (x1, y1), (x2, y2), (0, 255, 0), 5)

     cv2.imshow('frame', size)
     cv2.imshow('edges', edges)

     key = cv2.waitKey(25)
     if key == 27:
          break

video.release()
cv2.destroyAllWindows()
