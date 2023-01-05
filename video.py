import cv2
import os

image_folder = 'clip'
video_name = 'video.avi'

clip = [ img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder,clip[0]))
(height, width) = frame.shape[:2]
#video = cv2.VideoWriter(video_name, 0 , 1, (width,height))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'),10, (680,480))

for image in clip:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
