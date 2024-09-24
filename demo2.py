import sys
from corrector import *


cap = cv2.VideoCapture('src.mp4') # HD 1920 x 1080 @ 12.5Hz

if (cap.isOpened() == False): 
  print("Error opening video stream or file!")
  sys.exit()

print('Running, please wait...')

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", int(1920/3) , int(1080/3)) 

out = cv2.VideoWriter("out.mp4", fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=12.5, frameSize=(1920, 1080))

refs = np.array([[0, 0],
                 [200, 0],
                 [200, 140],
                 [0, 140]], dtype=np.float32) # the markers are spaced in a 14cmx20cm rectangle
        
q = Corrector(refs_mm=refs, 
              marker_ids=np.array([102, 104, 103, 101], dtype=np.int32), # from top left, moving clockwise
              pix_to_mm=0.2,                                             # 1 pixel = 0.2 mm
              im_size=np.array([1920, 1080]),                            # our camera resolution
              origin_offset_mm=np.array([60., 35.]))                     # printing offset ca 6cm x 3.5cm

while(True):
  ret, frame = cap.read()
  
  if ret == True:
    frame = q.prefilter(frame)
    centers = q.get_image_points(frame)
    frame = q.do_correction(frame, centers)
    out.write(frame) # Write out frame to video

    cv2.imshow('Video',frame)
    cv2.waitKey(1)

  else: 
    break

out.release()
cap.release()
cv2.destroyAllWindows()

print("bye...")