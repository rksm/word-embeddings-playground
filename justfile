set dotenv-load

export RUST_BACKTRACE := "full"

default:
    just --list

run:
    cargo run --release

dev:
    cargo watch -x run

DOCKER_IMAGE := "rust-cuda"

docker-build:
    docker build -t {{DOCKER_IMAGE}} .

docker-run:
    docker run --rm \
      -v $PWD/docker-target:/app/target \
      -v $PWD/data:/app/data \
      --gpus all -it {{DOCKER_IMAGE}}
    # docker run --rm --gpus all -it {{DOCKER_IMAGE}} bash
    # docker run --rm -it {{DOCKER_IMAGE}} bash

docker: docker-build docker-run
