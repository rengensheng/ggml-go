#!/bin/bash

# Build script for ggml-go with ggml

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Build the ggml library first
echo "Building ggml library..."
cd ggml
mkdir -p build
cd build
cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_ACCELERATE=OFF
make -j$(nproc)

# Go back to the main directory
cd ../../

# Build the Go application
echo "Building Go application..."
go build -o ggml-go .

echo "Build complete!"