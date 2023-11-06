FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Update default packages
RUN apt-get update

# Get Ubuntu packages
RUN apt-get install -y \
    build-essential \
    curl xz-utils pkg-config libssl-dev zlib1g-dev libtinfo-dev libxml2-dev

# Update new packages
RUN apt-get update

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

RUN apt install nvidia-utils-535 -y

RUN mkdir /app
WORKDIR /app
COPY . .

ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo add candle_core --features cuda
# RUN cargo build

CMD ["bash", "run.sh"]
