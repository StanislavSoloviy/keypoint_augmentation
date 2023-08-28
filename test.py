from ultralytics import YOLO
from ultralytics import settings
import albumentations as A
import random
model = YOLO('yolov8n-pose.pt')

import cv2

def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    transform = A.Compose([
        A.PadIfNeeded(min_height=800, min_width=600, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0], p=1),
        A.Affine(scale=(0.85, 0.85), rotate=(5, 5), translate_percent=(0.1, 0.1), p=1),
        A.HorizontalFlip(p=1),
        A.Perspective(scale=(0.05), p=1)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        keypoints = results[0].keypoints.data[0, :, :-1].tolist()
        transformed = transform(image=frame, keypoints=keypoints)
        transformed_image = transformed['image']
        transformed_keypoints = transformed['keypoints']

        for point in transformed_keypoints:
            x, y = map(int, point)
            cv2.circle(transformed_image, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow('Transformed Frame', transformed_image)
        out.write(transformed_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Укажите пути к входному и выходному видеофайлам
input_video_path = "1.mp4"
output_video_path = "augmen_video.mp4"

# Вызов функции для обработки видеофайла
process_video(input_video_path, output_video_path)