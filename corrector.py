import numpy as np
import cv2


class Corrector:
    def __init__(self, marker_ids, refs_mm, pix_to_mm, im_size, origin_offset_mm):
        self.setup_aruco_detector()

        self._marker_ids = marker_ids # from top left, moving clockwise
        self._pix_to_mm = pix_to_mm  # n pixel = (n * pix_to_mm) mm
        self._refs = (refs_mm + origin_offset_mm) / self._pix_to_mm # now in pixels

        self._im_w = im_size[0]
        self._im_h = im_size[1]

        
    def setup_aruco_detector(self, aruco_dict=cv2.aruco.DICT_6X6_1000):
        arucoDict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        arucoParams = cv2.aruco.DetectorParameters()
        arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self._aruco_detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)


    def get_aruco_detector(self):
        return self._aruco_detector


    def get_image_points(self, image_frame):
        corners, ids, _ = self._aruco_detector.detectMarkers(image_frame)
        cv2.aruco.drawDetectedMarkers(image_frame, corners, ids)
        centers = {}
        for k in range(len(corners)):
            u, v  = np.mean(corners[k][0],axis=0)
            centers[ids[k][0]] = np.array([u, v])
            xint, yint = int(round(u)), int(round(v))
            cv2.circle(image_frame, (xint, yint), radius=5, color=(0, 0, 255), thickness=-1)

        return centers
    

    def prefilter(self, image_frame):
        alpha = 1.5 # contrast control (1.0-3.0)
        beta = 0    # brightness control (0-100)
        image_frame = cv2.convertScaleAbs(image_frame, alpha=alpha, beta=beta)
        return image_frame


    def do_correction(self, image_frame, centers):
        if len(centers) == 4:
            srcs = np.vstack((centers[self._marker_ids[0]], 
                              centers[self._marker_ids[1]], 
                              centers[self._marker_ids[2]], 
                              centers[self._marker_ids[3]]))
            
            h, status = cv2.findHomography(srcs, self._refs)
            
            image_dst = cv2.warpPerspective(image_frame, h, [self._im_w, self._im_h])
        
        else :
            image_dst = np.zeros([self._im_h, self._im_w, 3], dtype=np.uint16)
        
        return image_dst