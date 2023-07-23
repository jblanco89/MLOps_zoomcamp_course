#!/bin/bash

apt-get update
apt-get install -y make


make create_data_directory

make set_data_directory_permissions

# Build the Docker image
make build_docker_image

make run_docker_container
