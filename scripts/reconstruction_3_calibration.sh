#!/bin/bash

source ../.env
colmap view_graph_calibrator --database_path "$COLMAP_PROJECT/database.db"
