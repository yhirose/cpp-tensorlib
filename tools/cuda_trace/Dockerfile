# The host toolchain the launch trace is built with. No CUDA toolkit: the
# backend's host side needs only a C++ compiler, and the driver is the fake.
FROM debian:trixie-slim
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends g++ python3 \
 && rm -rf /var/lib/apt/lists/*
