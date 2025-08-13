# ggml-go

A Go language wrapper for the ggml library, providing a PyTorch-like interface for tensor operations and computation graphs.

## Features

- Tensor operations with a familiar PyTorch-like API
- Context management for memory allocation
- Backend support for computation execution
- Computation graph building and execution

## Directory Structure

```
ggml-go/
├── ctx/         # Context management
├── tensor/      # Tensor operations
├── backend/     # Backend computation
├── ggml/        # ggml library as submodule
├── example.go   # Example usage
├── build.sh     # Build script
└── README.md    # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone --recursive https://github.com/rengensheng/ggml-go.git
   cd ggml-go
   ```

2. Build the project:
   ```bash
   ./build.sh
   ```

## Usage

See `example.go` for a basic example of how to use the library:

```go
// Initialize context
params := ggml.InitParams{
    MemSize:   1024 * 1024 * 1024, // 1GB
    MemBuffer: nil,
    NoAlloc:   false,
}
context := ctx.New(params)
defer context.Free()

// Create tensors
a := tensor.NewTensor1D(context, ggml.TypeF32, 10)
b := tensor.NewTensor1D(context, ggml.TypeF32, 10)

// Set values
for i := 0; i < 10; i++ {
    a.SetF32(i, float32(i+1))
    b.SetF32(i, float32(i+2))
}

// Perform addition
c := tensor.Add(context, a, b)

// Create and compute graph
graph := backend.NewGraph(context)
graph.BuildForwardExpand(c)
cpuBackend := backend.NewCPUBackend()
defer cpuBackend.Free()
status := cpuBackend.GraphCompute(graph)
```

## API Overview

### Context (`ctx` package)

- `New(params)` - Create a new context
- `Free()` - Free the context
- `UsedMem()` - Get memory usage
- `Reset()` - Reset the context

### Tensors (`tensor` package)

- `NewTensor1D/2D/3D/4D(ctx, type, dimensions...)` - Create tensors
- `Add/Sub/Mul/Div` - Element-wise operations
- `Sqr/Sqrt/Sum/Mean` - Mathematical operations
- `Reshape/Permute/Transpose` - Shape operations
- `SetF32/GetF32` - Access tensor data

### Backend (`backend` package)

- `NewCPUBackend()` - Create CPU backend
- `GraphCompute(graph)` - Execute computation graph
- `TensorSet/TensorGet` - Transfer data to/from tensors

## Building

Run the build script:

```bash
./build.sh
```

This will:
1. Build the ggml library
2. Build the Go application

## License

This project is licensed under the MIT License - see the LICENSE file for details.