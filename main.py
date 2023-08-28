import fastapi
from fastapi import FastAPI, File, UploadFile
import cv2
import os
from PIL import Image
import keypoints_augmentations
import shutil
from random import randint
import uvicorn
from zipfile import ZipFile

def create_random_dir() -> str:
    '''Функция для создания рандомного имени для папки'''
    result = 'temp_'
    for _ in range(10):
        result += chr(randint(97, 122))
    return result


app = FastAPI()

@app.post("/upload/")
async def upload_files(video_file: UploadFile = File(...)):
    # Генерируем случайное имя папки, проверяем, есть ли такая, если есть, выбираем другое имя, если нет, создаём
    UPLOAD_FOLDER = create_random_dir()
    while UPLOAD_FOLDER in os.listdir():
        UPLOAD_FOLDER = create_random_dir()

    if 'result.zip' in os.listdir():
        os.remove('result.zip')

    # Создаем папку
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    print(UPLOAD_FOLDER)

    # Сохраняем загруженные файлы
    file_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    with open(file_path, "wb") as f1:
        shutil.copyfileobj(video_file.file, f1)

    # Вызов функции
    keypoints_augmentations.kpts_augmen_coords(
        input_video_path=file_path,
        csv_save_path=UPLOAD_FOLDER + '\\result.csv',
        seq_len=5,
        augment=True,
        display= True,
    )


    with ZipFile('result.zip', mode='w') as zip_file:
        zip_file.write(UPLOAD_FOLDER + '\\result.mp4')
        zip_file.write(UPLOAD_FOLDER + '\\result.csv')

    shutil.rmtree(UPLOAD_FOLDER)


    return fastapi.responses.FileResponse(
        path="result.zip",
        filename='Результат.zip')

if __name__ == '__main__':
    uvicorn.run(app)
