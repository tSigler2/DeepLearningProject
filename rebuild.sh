#!/bin/bash

IMAGE_NAME="nb_dl_project"
CONTAINER_NAME="nb_dl_container"

if [ "$(docker ps -aq -f name=^${CONTAINER_NAME}$)" ]; then
    echo "Stopping and removing existing container..."
    docker stop ${CONTAINER_NAME}
    docker rm ${CONTAINER_NAME}
fi

echo "Building Docker image..."
docker build -t ${IMAGE_NAME} .

echo "Running new Docker container..."
docker run -it --name ${CONTAINER_NAME} \
  -p 8888:8888 \
  -v "$PWD":/usr/src/app \
  ${IMAGE_NAME}