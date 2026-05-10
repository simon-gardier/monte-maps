#!/bin/bash

source ../.env
colmap feature_extractor \
    --database_path "$COLMAP_PROJECT/database.db" \
    --image_path "$IMAGES_FOLDER" \
    --ImageReader.camera_model OPENCV \
    --ImageReader.single_camera 1 \
    --SiftExtraction.max_num_features 8192 \
    --FeatureExtraction.use_gpu 1
