FROM tensorflow/tensorflow:latest-py3

# Install system packages
RUN apt-get update

COPY . /app

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt


CMD ["python", "train.py"]