# YOLOv3 🚀 by Ultralytics, GPL-3.0 license

# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.10-py3

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install --no-cache -r requirements.txt coremltools onnx gsutil notebook wandb>=0.12.2
#RUN pip install --no-cache -U torch torchvision numpy Pillow
RUN pip install --no-cache torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Create working directory
#RUN mkdir -p /usr/src/yolov3
#WORKDIR /usr/src/yolov3

# Requirements
#COPY ./requirements.txt /usr/src/yolov3/requirements.txt
#RUN pip install --no-cache -r requirements.txt

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf /root/.config/Ultralytics/

# Set environment variables
# ENV HOME=/usr/src/yolov3
# ENV PYTHONPATH=$PYTHONPATH:/usr/src/

# Install ROS Noetic
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y curl lsb-release
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt update && apt install -y ros-noetic-ros-base
# pip install rospkg: probably not the best solution, but seems to work:
RUN pip install rospkg
#RUN pip install cvbridge3
# installing tf dependency here. Also not the best idea. (package.xml doesn't exist here)
RUN apt update && apt install -y ros-noetic-tf
ENV PYTHONPATH=$PYTHONPATH:/opt/ros/noetic/lib/python3/dist-packages
# ENV PYTHONPATH=$PYTHONPATH:/usr/src/

# Copy contents
#COPY . /usr/src/yolov3

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute,display

#entrypoint is always run at container startup
COPY entrypoint.sh /root/
#ENTRYPOINT ["/root/entrypoint.sh"]
#CMD ["python" "./pose_estimate/inference.py"]
CMD ["bash", "-l"]

# Usage Examples -------------------------------------------------------------------------------------------------------

# Build and Push
# t=ultralytics/yolov3:latest && sudo docker build -t $t . && sudo docker push $t

# Pull and Run
# t=ultralytics/yolov3:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all $t

# Pull and Run with local directory access
# t=ultralytics/yolov3:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/datasets:/usr/src/datasets $t

# Kill all
# sudo docker kill $(sudo docker ps -q)

# Kill all image-based
# sudo docker kill $(sudo docker ps -qa --filter ancestor=ultralytics/yolov3:latest)

# Bash into running container
# sudo docker exec -it 5a9b5863d93d bash

# Bash into stopped container
# id=$(sudo docker ps -qa) && sudo docker start $id && sudo docker exec -it $id bash

# Clean up
# docker system prune -a --volumes

# Update Ubuntu drivers
# https://www.maketecheasier.com/install-nvidia-drivers-ubuntu/

# DDP test
# python -m torch.distributed.run --nproc_per_node 2 --master_port 1 train.py --epochs 3
