import cv2

frame_path = '/Users/marinaalonsopoal/Desktop/Tracking/Datasets/SMOT/acrobats/img/000057.jpg'

image = cv2.imread(frame_path)

xleft, ytop, xright, ybottom = 435, 253, 483, 385

# Display
cv2.imshow("output", image)

fromCenter = False
r = cv2.selectROI("Image", image, fromCenter)

print('\nCoordinates:')
print(r, '\n')

cv2.destroyAllWindows()
