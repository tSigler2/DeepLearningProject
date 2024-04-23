# Study of the Adaptability of ResNet50 to new Images with Disparate Context

## Overview
Our project utilizes Docker, JupyterLab, and standard Python to provide multiple avenues to run our project.

## Prerequisites
- Docker Installed on Your Machine [Get Docker](https://docs.docker.com/get-docker/)
- Understanding of the Command Line and Docker

## Setting Up Docker

### Building Docker Image
Building the Docker image just requires running the `build.sh` script. This script will create an image name `nb_dl_project` and starts a container with this image named `nb_dl_container`.

```bash
./build.sh
```

### Rebuilding the Docker Environment
Rebuilding the Docker container uses the `rebuild.sh` script. This script removes the existing container, rebuilds the image, and starts a new container with this rebuilt image.

```bash
./rebuild.sh
```


## Using the Project
### Accessing JupyterLab
After running `build.sh` or `rebuild.sh`, JupyterLab will be available at `http://localhost:8888`. The command line will provide a token for the session.

### Restarting the Same Container
If you want to restart the container without making any changes to the Docker image, you can simply start the container again. You can do this with the following Docker command:

```
bash 
docker start -ai nb_dl_project
```

## Running with Python
This project contains a `.py` file that the user can run with their local Python interpreter. This project was written targetting Python 3.9.7.

Command `/usr/local/bin/python3 ./DeeplearningProject/test_proj.py`

## Hyper-Parameter Tuning with RayTune
Due to how RayTune runs its testing environment, hyper-parameter tuning requires that all relative paths be replaced with their absolute path for the program to run. The relative paths have been left in for users so that they can fill in with the path to this folder on their computer.