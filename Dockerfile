FROM tensorflow/tensorflow:latest-py3

# Install system packages
RUN apt-get update

COPY src /src

WORKDIR /src

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV PYTHONPATH='/src/:$PYTHONPATH'

CMD ["python", "train.py"]