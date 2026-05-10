#!/bin/bash

source ../.env
mkdir -p "$COLMAP_PROJECT/sparse"
colmap global_mapper \
    --database_path "$COLMAP_PROJECT/database.db" \
    --image_path "$IMAGES_FOLDER" \
    --output_path "$COLMAP_PROJECT/sparse"
