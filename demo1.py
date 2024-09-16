from corrector import *

cap = cv2.VideoCapture('src.mp4')


cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", int(1920/3) , int(1080/3)) 

q = Corrector()


if (cap.isOpened()== False): 
    print("Error opening video stream or file")
 
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter("out.mp4", fourcc, 12.5, (1920, 1080))


while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    centers = q.get_image_points(frame, aruco_size=50)
    frame = q.do_correction(frame, centers)
    
    cv2.imshow('Video',frame)

    out.write(frame) # Write out frame to video


    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  else: 
    break
 
out.release()
cap.release()
cv2.destroyAllWindows()

