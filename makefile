# Define variables
DOCKER_IMAGE_NAME = lstm_app_image
CURRENT_DIR := $(shell pwd)
DATA_DIR = data

# Create the directory if it does not exist
create_data_directory:
        @if [ ! -d "$(DATA_DIR)" ]; then \
                mkdir -p "$(DATA_DIR)"; \
        fi

set_data_directory_permissions:
        chmod -R 777 "$(DATA_DIR)"

build_docker_image:
        docker build -t $(DOCKER_IMAGE_NAME) .

run_docker_container:
        docker run -p 5000:5000 $(DOCKER_IMAGE_NAME)

# Phony targets (these targets are not file names)
.PHONY: create_data_directory set_data_directory_permissions build_docker_image run_docker_container
