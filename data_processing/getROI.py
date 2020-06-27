import cv2

frame_path = '/Users/marinaalonsopoal/Desktop/Tracking/Datasets/SMOT/acrobats/img/000037.jpg'
frame_path = '/Users/marinaalonsopoal/Desktop/Tracking/Datasets/eSMOT/football/img/00055.jpeg'

image = cv2.imread(frame_path)

# Display
cv2.imshow("output", image)

fromCenter = False
r = cv2.selectROI("Image", image, fromCenter)

xleft, ytop, xright, ybottom = r
print('r:', r)


cv2.destroyAllWindows()
