#!/bin/bash

# Nombre y tag de la imagen que quieres usar
IMAGE_NAME="miusuario/miimagen:latest"

# Path del dataset (ajusta si es necesario)
DATASET_PATH="/"

# Nombre del contenedor
CONTAINER_NAME="birds-classification"

# Recursos
SHM_SIZE="16gb"
MEMORY="24gb"

# GPU que quieres usar (ajusta si es necesario)
GPU_DEVICE=1

# 1. Si la imagen NO existe, constrúyela
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "Imagen $IMAGE_NAME no encontrada. Construyendo la imagen..."
    docker build -t $IMAGE_NAME .
    if [[ $? -ne 0 ]]; then
        echo "La construcción de la imagen falló. Abortando."
        exit 1
    fi
fi

# 2. Corre el contenedor
docker run --gpus "\"device=${GPU_DEVICE}\"" --rm -it \
    --name ${CONTAINER_NAME} \
    -v ${DATASET_PATH}:/data/:ro \
    -v "$(pwd)"/:/workspace/:rw \
    --shm-size=${SHM_SIZE} \
    --memory=${MEMORY} \
    ${IMAGE_NAME} ./run.sh
