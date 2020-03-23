import cv2

path = '/Users/marinaalonsopoal/Desktop/'
video_name = 'caminant'
video = path + video_name + '.mp4'
vidcap = cv2.VideoCapture(video)
success, image = vidcap.read()
count = 0

while success:
    frame_name = str(count)
    frame_name = path + video_name + '/frame' + frame_name.zfill(3) + '.jpg'
    cv2.imwrite(frame_name, image)     # save frame as JPEG file
    success, image = vidcap.read()
    count += 1
