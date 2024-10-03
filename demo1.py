import sys
from corrector import *
import configparser, os

print(cv2.__version__)

config = configparser.ConfigParser()
config.read_file(open('./default.cfg'))
 
aruco_size      = float(config['DEFAULT']['aruco_size'])
origin_left     = float(config['DEFAULT']['origin_left'])
origin_top      = float(config['DEFAULT']['origin_top'])
imsize_w        = int(config['DEFAULT']['imsize_w'])
imsize_h        = int(config['DEFAULT']['imsize_h'])
pix_to_mm       = float(config['DEFAULT']['pix_to_mm'])

cv2.namedWindow('Photo', cv2.WINDOW_NORMAL)
cv2.resizeWindow("Photo", int(1920/3) , int(1080/3)) 

frame = cv2.imread("./img2.jpg")

N = aruco_size / 50.0 # actual size of the marker is 50mmx50mm

refs = N * np.array([[0  , 0],
                     [200, 0],
                     [200, 100],
                     [0  , 100]], dtype=np.float32) # the markers are spaced in a 14cmx20cm rectangle
        
q = Corrector(refs_mm=refs, 
              marker_ids=np.array([101, 102, 104, 103], dtype=np.int32), # from top left, moving clockwise
              pix_to_mm=pix_to_mm,                                           # 1 pixel = 0.2 mm
              im_size=np.array([imsize_w, imsize_h]),                            # our camera resolution
              origin_offset_mm=np.array([origin_left, origin_top]))                     # printing offset ca 6cm x 3.5cm

frame = q.prefilter(frame)
centers = q.get_image_points(frame)
frame = q.do_correction(frame, centers)

cv2.imwrite("./img2_.jpg", frame)
cv2.imshow('Photo',frame)
cv2.waitKey(0)

 

cv2.destroyAllWindows()

print("bye...")