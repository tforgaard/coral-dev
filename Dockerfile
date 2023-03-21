FROM python:slim-buster

RUN apt-get update && \
    apt-get install -y \
        git python3-dev gcc

RUN git clone https://github.com/crosstyan/mpu6050-py.git \
    && cd mpu6050-py \
    && pip install -e . \
    && cd ..

RUN git clone --recursive https://github.com/pimoroni/vl53l5cx-python.git \
    && cd vl53l5cx-python/library \
    && pip install -e . \
    && cd ../..

RUN git clone https://github.com/lefuturiste/BMI160-i2c.git \
    && cd BMI160-i2c \
    && pip install -e . \
    && cd ..

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ /models
COPY data /data
COPY src/ .

ENV UDEV=1

ENTRYPOINT [ "/bin/sh" ]

# ENTRYPOINT [ "python", "serialrpc.py" ]
