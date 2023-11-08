set dotenv-load

export RUST_BACKTRACE := "full"

default:
    just --list

run:
    # cargo run --release
    # cargo run --release -- -f data/w2v_skipgram.nn --method skip-gram
    # cargo run --release -- --method skip-gram
    # cargo run --release --example embeddings-cbow
    cargo run --release -- --method cbow

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

conda:
    conda activate word-embeddings

jupyter:
    jupyter notebook --no-browser

    # jupyter notebook --no-browser --port-retries=0 --port=8888 --ip=0.0.0.0 \
    #   --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.disable_check_xsrf=True
