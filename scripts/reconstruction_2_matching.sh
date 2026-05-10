#!/bin/bash

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 
source ../.env
colmap sequential_matcher \
    --database_path "$COLMAP_PROJECT/database.db" \
    --FeatureMatching.guided_matching 1 \
    --FeatureMatching.use_gpu 1 \
    --SequentialMatching.overlap 10 \
    --SequentialMatching.loop_detection 1 \
    --SequentialMatching.vocab_tree_path "$VOCAB_TREE_PATH" \
    --SequentialMatching.loop_detection_period 5 \
    --SequentialMatching.loop_detection_num_images 100
