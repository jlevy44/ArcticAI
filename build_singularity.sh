#!bin/bash
docker build -t arcticai:latest .
sudo singularity build arcticai.sif docker-daemon://arcticai:latest