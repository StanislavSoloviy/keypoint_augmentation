FROM python:3.9

MAINTAINER Solsa

WORKDIR /app

COPY main.py /app/main.py
COPY keypoints_augmentations.py /app/keypoints_augmentations.py
COPY yolov8n-pose.pt /app/yolov8n-pose.pt
COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install -r requirements.txt

CMD ["python", "main.py"]
EXPOSE 8000