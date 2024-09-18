import sys
from corrector import *

from scipy.interpolate import CubicSpline


cap = cv2.VideoCapture('src.mp4') # HD 1920 x 1080 @ 12.5Hz

if (cap.isOpened() == False): 
  print("Error opening video stream or file!")
  sys.exit()

print('Running, please wait...')

'''
First, we extract all frames and the image points
'''
C = []
k = 0
frames = []

q = Corrector()

while(True):
  ret, frame = cap.read()
  
  if ret == True:
    frame = q.prefilter(frame)
    centers = q.get_image_points(frame, aruco_size=50)

    frames.append(frame)

    if len(centers) == 4:
        C.append(np.hstack((k, 
                            centers[101],  
                            centers[102],  
                            centers[103],  
                            centers[104])))

  else: 
    break

  k = k + 1 

cap.release()

'''
Next, we build an interpolation funtion to accomodate 
missing/undetected image points.
'''
C = np.array(C)
f101_x = CubicSpline(C[:,0], C[:,1], bc_type='clamped')
f101_y = CubicSpline(C[:,0], C[:,2], bc_type='clamped')
f102_x = CubicSpline(C[:,0], C[:,3], bc_type='clamped')
f102_y = CubicSpline(C[:,0], C[:,4], bc_type='clamped')
f103_x = CubicSpline(C[:,0], C[:,5], bc_type='clamped')
f103_y = CubicSpline(C[:,0], C[:,6], bc_type='clamped')
f104_x = CubicSpline(C[:,0], C[:,7], bc_type='clamped')
f104_y = CubicSpline(C[:,0], C[:,8], bc_type='clamped')

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", int(1920/3) , int(1080/3))

out = cv2.VideoWriter("out.mp4", fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=12.5, frameSize=(1920, 1080))

centers 
for k in range(len(frames)):
    centers[101] = np.array([f101_x(k), f101_y(k)])
    centers[102] = np.array([f102_x(k), f102_y(k)])
    centers[103] = np.array([f103_x(k), f103_y(k)])
    centers[104] = np.array([f104_x(k), f104_y(k)])

    frame = q.do_correction(frames[k], centers)
    out.write(frame) # Write out frame to video

    cv2.imshow('Video',frame)
    cv2.waitKey(1)

out.release()
cv2.destroyAllWindows()

print("bye...")