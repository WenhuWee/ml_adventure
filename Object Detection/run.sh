#!/bin/bash

# export PYTHONPATH=$PYTHONPATH:/tf_ob/tf_models:/tf_object_detection/tf_models/research:/tf_object_detection/tf_models/research/slim
export PYTHONPATH=$PYTHONPATH:/tf_ob:/tf_ob/slim
export PATH=$PATH:$PYTHONPATH

# python /tf_object_detection/tf_models/research/setup.py build
# python /tf_object_detection/tf_models/research/setup.py install

# python /tf_object_detection/research/object_detection/train.py --logtostderr --train_dir=/output/ --pipeline_config_path=./faster_rcnn_resnet101_coco.config

# python /tf_object_detection/tf_models/research/object_detection/model_main.py --alsologtostderr --model_dir=/output --pipeline_config_path=./pipeline_floyd.config

python /tf_ob/object_detection/model_main.py --alsologtostderr --model_dir=/output --pipeline_config_path=./pipeline_floyd.config
