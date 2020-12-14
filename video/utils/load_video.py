import os
import numpy as np
import cv2

def load_video(path: str) -> np.ndarray:
    ext_list = [".mp4", ".avi", ".mov"]
    video_ext = os.path.splitext(os.path.basename(path))[1].lower()
    if video_ext in ext_list:
        clean_video_path = path
    else:
        ValueError(f"{video_ext} is not supported.")

    cap_video = cv2.VideoCapture(clean_video_path)
    frame_length = cap_video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap_video.get(cv2.CAP_PROP_FPS)

    print("frame length:", int(frame_length))
    print("fps:", int(fps))

    frames = []
    while cap_video.isOpened():
        ret, frame = cap_video.read()
        if not ret:
            break

        frames.append(frame)
        if len(frames) == frame_length:
            break
    cap_video.release()

    return np.array(frames)
