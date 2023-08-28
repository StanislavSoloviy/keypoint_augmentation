import argparse
import cv2
import albumentations as A
import random
from ultralytics import YOLO
from collections import deque
import csv
import glob
import os


def kpts_augmen_coords(input_video_path: str,
                       csv_save_path: str,
                       seq_len: int,
                       augment: bool,
                       scale_range: tuple = (0.85, 1.25),
                       rotate_range: tuple = (-5, 5),
                       translate_range: tuple = (-0.00, 0.00),
                       perspective_scale_range: tuple = (0.01, 0.05),
                       model_name: str = 'yolov8n-pose.pt',
                       display: bool = True,
                       ) -> None:
    """
    Покадровая аугментация ключевых точек. Набор аугментаций фиксированный: аффинные преобразования
    (масштабирование, сдвиг и поворот), горизонтальное отражение, четырехточечное изменение перспективы. Параметры
    аугментаций выбираются случайно из заданных диапазонов значений, которые отражают приемлемое (сохранение ударов
    бойцов) преобразование кадра. Последующая запись координат ключевых точек в csv файл с заданной длиной
    последовательности.

    Args:
        input_video_path (str): Путь к входному видеофайлу.
        csv_save_path (str): Путь для сохранения CSV-файла с данными ключевых точек.
        seq_len (int): Длина последовательности координат ключевых точек.
        augment (bool): Флаг включения аугментации.
        scale_range (tuple): Диапазон масштабирования.
            (default is (0.85, 1.15))
        rotate_range (tuple): Диапазон поворота в градусах.
            (default is (-5, 5))
        translate_range (tuple): Диапазон трансляции в процентных отношениях.
            (default is (-0.05, 0.05))
        perspective_scale_range (tuple): Диапазон перспективы.
            (default is (0.01, 0.05))
        model_name (str): Модель.
            (default is 'yolov8n-pose.pt')
        display (bool): Отображение аугментированных кадров с ключевыми точками.
            (default is True)
    """
    # Параметры для видео
    cap = cv2.VideoCapture(input_video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_video_path = os.path.dirname(input_video_path) + "\\result.mp4"

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    # случайный выбор параметров аугментации в заданных диапазонах
    scale = random.uniform(scale_range[0], scale_range[1])
    rotate = random.uniform(rotate_range[0], rotate_range[1])
    translate = random.uniform(translate_range[0], translate_range[1])
    flip_p = random.choice([0, 1])
    perspective_scale = random.uniform(perspective_scale_range[0], perspective_scale_range[1])

    # пайплайн аугментаций с выбранными параметрами
    if augment:
        transform = A.Compose([
            A.PadIfNeeded(min_height=640, min_width=480, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0], p=1),
            A.Affine(scale=(scale, scale), rotate=(rotate, rotate), translate_percent=(translate, translate), p=1),
            A.HorizontalFlip(p=flip_p),
            A.Perspective(scale=(0, perspective_scale), p=1)
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    else:
        transform = A.Compose([
            A.NoOp()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    # создание экземпляра класса предобученной модели YOLO
    model = YOLO(model_name)
    #model.to('cuda')

    csv_file = open(csv_save_path, 'w', newline='')  # Открытие CSV файла
    writer = csv.writer(csv_file)

    data_buffer = deque(maxlen=seq_len)  # циклический буфер записи
    cap = cv2.VideoCapture(input_video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        # аугментация ключевых точек на кадре
        keypoints = results[0].keypoints.data[0, :, :-1].tolist()
        transformed = transform(image=frame, keypoints=keypoints)
        augmen_keypoints = transformed['keypoints']
        augmen_keypoints_lst = [[x, y] for x, y in augmen_keypoints]
        data_buffer.append(augmen_keypoints_lst)

        # запись в csv файл
        if len(data_buffer) == seq_len:
            writer.writerow(data_buffer)

        # отображение аугментированных кадров с ключевыми точками
        augmen_image = transformed['image']
        for point in augmen_keypoints:
            x, y = map(int, point)
            cv2.circle(augmen_image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Transformed Frame', augmen_image)
        out.write(augmen_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()
    csv_file.close()



if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Augment keypoints in a video using YOLO model')
    # parser.add_argument('--input_video', type=str, required=True, help='Path to input video file')
    # parser.add_argument('--csv_save_path', type=str, required=True, help='Path to save the CSV file with the '
    #                                                                      'coordinates of the key points '
    #                                                                      '(path/file_name.csv)')
    # parser.add_argument('--model_name', type=str, default='yolov8n-pose.pt', help='YOLO model')
    # parser.add_argument('--seq_len', type=int, required=True, help='Keypoint coordinate sequence length')
    # parser.add_argument('--augment', action='store_true', help='Augment enable flag')
    # parser.add_argument('--display', action='store_true', help='Display augmented frames and keypoints')
    # args = parser.parse_args()
    #
    # kpts_augmen_coords(
    #     args.input_video,
    #     args.csv_save_path,
    #     args.seq_len,
    #     model_name=args.model_name,
    #     augment=args.augment,
    #     display=args.display
    # )

    kpts_augmen_coords(
        input_video_path = "1.mp4",
        csv_save_path= "result.csv",
        seq_len= 5,
        augment= True,
        scale_range = (0.85, 1.25),
        rotate_range = (-5, 5),
        translate_range= (-0.00, 0.00),
        perspective_scale_range = (0.01, 0.05),
        display = True,
        )



