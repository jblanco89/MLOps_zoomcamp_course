#!/bin/bash

# updates packages, installs the make utility, 
# creates and sets permissions for a data directory, 
# builds a Docker image, and runs a Docker container for an LSTM application.

apt-get update
apt-get install -y make


make create_data_directory

make set_data_directory_permissions

# Build the Docker image
make build_docker_image

make run_docker_container
