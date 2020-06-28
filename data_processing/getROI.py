import cv2

# frame_path = '/Users/marinaalonsopoal/Desktop/Tracking/Datasets/SMOT/juggling/img/000022.jpg'
frame_path = '/Users/marinaalonsopoal/Desktop/Tracking/Datasets/eSMOT/bolt1/img/00000024.jpg'

image = cv2.imread(frame_path)

# Display
cv2.imshow("output", image)

fromCenter = False
r = cv2.selectROI("Image", image, fromCenter)

xleft, ytop, xright, ybottom = r
print('r:', r)


cv2.destroyAllWindows()
