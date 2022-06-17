#!/bin/bash

# Note: Dockerfile achieves the same thing without this script
echo "Fetching weights..."
mkdir -p weights
wget --no-clobber https://github.com/robberthofmanfm/yolo/releases/download/v0.0.1/obj19_detector.pt -O weights/detector.pt
wget --no-clobber https://github.com/robberthofmanfm/yolo/releases/download/v0.0.1/encoder.npy -O weights/encoder.npy
wget --no-clobber https://github.com/robberthofmanfm/yolo/releases/download/v0.0.1/obj_19_pose_estimator_model-epoch199.pt -O weights/pose_estimator.pt
echo "Weights should be in ./weights folder"

echo "Adding home folder to pythonpath - TODO should not be necessary"
export PYTHONPATH=$PYTHONPATH:~
echo "Ok - might want to add following line to .bashrc as well:"
echo 'export PYTHONPATH=$PYTHONPATH:~'


echo '=== End of install script ==='

