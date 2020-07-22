FROM tensorflow/tensorflow:1.15.2

WORKDIR /pepper_tensorflow
COPY pepper_tensorflow .

CMD ["python", "object_detection.py"]

EXPOSE 27001 27002 27003
