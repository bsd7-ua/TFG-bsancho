FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Instalar dependencias del sistema necesarias (incluyendo las librerías para OpenCV)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo
WORKDIR /home/bsancho/workspace/TAS

# Copiar el archivo de dependencias y luego instalarlas
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Instalar dependencias adicionales
RUN pip3 install torch==1.13.1 torchvision==0.14.1 timm opencv-python tqdm pandas

# Copiar los scripts necesarios en el contenedor
COPY extract_and_train.py .
COPY run.sh .

# Hacer ejecutable el run.sh
RUN chmod +x run.sh

# Comando por defecto: ejecutar el run.sh
CMD ["./run.sh"]
