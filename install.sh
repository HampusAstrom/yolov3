#!/bin/bash

# Note: Dockerfile achieves the same thing without this script
echo "Fetching weights..."
mkdir -p weights
wget https://github.com/robberthofmanfm/yolo/releases/download/v0.0.1/obj19_detector.pt -o weights/detector.pt
wget https://github.com/robberthofmanfm/yolo/releases/download/v0.0.1/encoder.npy -o weights/encoder.npy
wget https://github.com/robberthofmanfm/yolo/releases/download/v0.0.1/obj_19_pose_estimator_model-epoch199.pt -o weights/pose_estimator.pt
echo "Weights should be in ./weights folder"

echo "Adding home folder to pythonpath - TODO should not be necessary"
export PYTHONPATH=$PYTHONPATH:~
echo "Ok - might want to add following line to .bashrc as well:"
echo 'export PYTHONPATH=$PYTHONPATH:~'


echo '=== End of install script ==='

