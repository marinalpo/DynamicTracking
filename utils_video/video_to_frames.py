import cv2

path = '/Users/marinaalonsopoal/Desktop/stanford_campus_dataset/videos/'
video_name = 'quad'
video = path + video_name + '/video0/video.mov'
vidcap = cv2.VideoCapture(video)
success, image = vidcap.read()
count = 0
num_frames = 300

while count < num_frames:
    frame_name = str(count)
    frame_name = path + video_name + '/frames0/' + frame_name.zfill(6) + '.jpg'
    cv2.imwrite(frame_name, image)     # save frame as JPEG file
    success, image = vidcap.read()
    count += 1
