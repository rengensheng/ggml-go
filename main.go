package main

/*
#cgo CFLAGS: -I${SRCDIR}/ggml/include
#cgo LDFLAGS: -L${SRCDIR}/ggml/build/src -lggml -lggml-base -lggml-cpu -lc++
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
*/
import (
	"fmt"

	"github.com/rengensheng/ggml-go/backend"
	"github.com/rengensheng/ggml-go/ctx"
	"github.com/rengensheng/ggml-go/lm"
	"github.com/rengensheng/ggml-go/tensor"
)

func main() {
	// Initialize context
	params := lm.InitParams{
		MemSize:   1024 * 1024 * 1024, // 1GB
		MemBuffer: nil,
		NoAlloc:   false,
	}
	context := ctx.New(params)
	defer context.Free()

	// Create a CPU backend
	cpuBackend := backend.NewCPUBackend()
	defer cpuBackend.Free()

	fmt.Printf("Using backend: %s\n", cpuBackend.Name())

	// Create tensors
	a := tensor.NewTensor1D(context, lm.TypeF32, 10)
	b := tensor.NewTensor1D(context, lm.TypeF32, 10)

	// Set values in tensors
	for i := 0; i < 10; i++ {
		a.SetF32(i, float32(i+1))
		b.SetF32(i, float32(i+2))
	}

	// Perform addition
	c := tensor.Add(context, a, b)

	// Create computation graph
	graph := backend.NewGraph(context)
	graph.BuildForwardExpand(c)

	// Compute the graph
	status := cpuBackend.GraphCompute(graph)
	if status != lm.StatusSuccess {
		fmt.Printf("Computation failed with status: %d\n", status)
		return
	}

	// Print results
	fmt.Println("Result of addition:")
	for i := 0; i < 10; i++ {
		fmt.Printf("c[%d] = %.2f\n", i, c.GetF32(i))
	}
}
