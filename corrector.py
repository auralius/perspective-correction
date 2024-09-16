import numpy as np
import cv2


class Corrector:
    def __init__(self):
        self.setup_aruco_detector()

        self._pix_to_mm = 0.2  # 1 pixel = 0.2 mm
        self._refs = np.array([[0, 0],
                               [200, 0],
                               [200, 140],
                               [0, 140]], dtype=np.float32) # the markers are spaced in a 14cmx20cm rectangle
        
        self._refs = (self._refs + np.array([60., 35.])) / self._pix_to_mm # printing offset ca 6cm x 3.5cm

        self._mmw = 1920
        self._mmh = 1080

        
    def setup_aruco_detector(self, aruco_dict=cv2.aruco.DICT_6X6_1000):
        arucoDict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        arucoParams = cv2.aruco.DetectorParameters()
        arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._aruco_detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)


    def get_aruco_detector(self):
        return self._aruco_detector


    def get_image_points(self, image_frame, aruco_size=50):
        corners, ids, _ = self._aruco_detector.detectMarkers(image_frame)
        cv2.aruco.drawDetectedMarkers(image_frame, corners, ids)
        centers = {}
        for k in range(len(corners)):
            u, v  = np.mean(corners[k][0],axis=0)
            centers[ids[k][0]] = np.array([u, v])
            xint, yint = int(round(u)), int(round(v))
            cv2.circle(image_frame, (xint, yint), radius=5, color=(0, 0, 255), thickness=-1)

        return centers
    

    def do_correction(self, image_frame, centers):
        if len(centers) == 4:
            srcs = np.vstack((centers[102], 
                              centers[104], 
                              centers[103], 
                              centers[101]))
            
            h, status = cv2.findHomography(srcs, self._refs)
            
            image_dst = cv2.warpPerspective(image_frame, h, [self._mmw, self._mmh])
        
        else :
            image_dst = np.zeros([self._mmh, self._mmw, 3], dtype=np.uint16)
        
        return image_dst