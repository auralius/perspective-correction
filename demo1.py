import sys
from corrector import *


cv2.namedWindow('Photo', cv2.WINDOW_NORMAL)
cv2.resizeWindow("Photo", int(1920/3) , int(1080/3)) 

frame = cv2.imread("./img2.jpg")

refs = np.array([[0, 0],
                 [200, 0],
                 [200, 100],
                 [0, 100]], dtype=np.float32) # the markers are spaced in a 14cmx20cm rectangle
        
q = Corrector(refs_mm=refs, 
              marker_ids=np.array([101, 102, 104, 103], dtype=np.int32), # from top left, moving clockwise
              pix_to_mm=0.103,                                           # 1 pixel = 0.2 mm
              im_size=np.array([3120, 4160]),                            # our camera resolution
              origin_offset_mm=np.array([60., 25.]))                     # printing offset ca 6cm x 3.5cm

frame = q.prefilter(frame)
centers = q.get_image_points(frame)
frame = q.do_correction(frame, centers)

cv2.imwrite("./img2_.jpg", frame)
cv2.imshow('Photo',frame)
cv2.waitKey(0)

 

cv2.destroyAllWindows()

print("bye...")