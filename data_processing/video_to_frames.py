import cv2
import os

path = '/Users/marinaalonsopoal/Desktop/Tracking/Datasets/stanford-campus/videos/'
sequences = sorted(os.listdir(path))
for s in range(len(sequences)):
    print('Sequence:', sequences[s])
    videos = sorted(os.listdir(path + sequences[s] + '/'))
    for v in range(len(videos)):
        video_path = path + sequences[s] + '/' + videos[v] + '/'
        print('Video:', videos[v])
        vidcap = cv2.VideoCapture(video_path + 'video.mov')
        success, image = vidcap.read()
        count = 1
        while success:
            frame_name = str(count)
            frame_name = video_path + str(count).zfill(6) + '.jpg'
            cv2.imwrite(frame_name, image)  # save frame as JPEG file
            success, image = vidcap.read()
            count += 1
