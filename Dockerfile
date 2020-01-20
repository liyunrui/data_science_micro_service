FROM python:3.6.5

RUN mkdir -p /app
COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

# the container will listen on 5000
EXPOSE 5000

# saving training logs and serving logs, we'll mount the resoult to local folder
RUN mkdir logs