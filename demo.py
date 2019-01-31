import cv2,sys
from optical_flow.main import optical_flow
import numpy as np

path = sys.argv[1]
save_path = sys.argv[2]

if path == '-1':
	path = int(0)

video_capture = cv2.VideoCapture(path)
video_capture.set(3, 340)
video_capture.set(4, 480)
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter(save_path,fourcc, 6, (960,680))

index = 0

while True:
	ret,frame = video_capture.read()
	if ret is True:
		if index == 0:
			last_img = frame
		else:
			output = optical_flow(last_img,frame,window=51,stride=5,grad_th=0.1,low=5,high=10,arrow_length=10)
			last_img = frame
			save_frame = cv2.resize(output,(960,680), interpolation=cv2.INTER_CUBIC)
			out.write(save_frame)
			cv2.imshow('',output)
			k = cv2.waitKey(1)
			if k == ord('q'):
				break
	else:
		break
	index += 1

out.release()
video_capture.release()
cv2.destroyAllWindows()