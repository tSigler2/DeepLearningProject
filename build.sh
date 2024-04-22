#!/bin/bash

docker build -t nb_dl_project .

docker run -it --name nb_dl_container \
  -p 8888:8888 \
  -v "$PWD":/usr/src/app \
  nb_dl_project