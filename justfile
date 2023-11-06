set dotenv-load

export RUST_BACKTRACE := "1"

default:
    just --list

run:
    cargo run

dev:
    cargo watch -x run

DOCKER_IMAGE := "rust-cuda"

docker-build:
    docker build -t {{DOCKER_IMAGE}} .

docker-run:
    docker run --rm -v $PWD/docker-target:/app/target --gpus all -it {{DOCKER_IMAGE}} bash
    # docker run --rm --gpus all -it {{DOCKER_IMAGE}} bash
    # docker run --rm -it {{DOCKER_IMAGE}} bash
