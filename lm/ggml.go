package lm

/*
#cgo CFLAGS: -I${SRCDIR}/../ggml/include
#cgo LDFLAGS: -L${SRCDIR}/../ggml/build/src -lggml -lggml-base -lggml-cpu -lc++
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
*/
import "C"
import (
	"unsafe"
)

// Status represents the status of ggml operations
type Status int

const (
	StatusAllocFailed Status = -2
	StatusFailed      Status = -1
	StatusSuccess     Status = 0
	StatusAborted     Status = 1
)

// Type represents the data type of a tensor
type Type int

const (
	TypeF32  Type = 0
	TypeF16  Type = 1
	TypeQ4_0 Type = 2
	TypeQ4_1 Type = 3
	// TypeQ4_2 Type = 4 support has been removed
	// TypeQ4_3 Type = 5 support has been removed
	TypeQ5_0    Type = 6
	TypeQ5_1    Type = 7
	TypeQ8_0    Type = 8
	TypeQ8_1    Type = 9
	TypeQ2_K    Type = 10
	TypeQ3_K    Type = 11
	TypeQ4_K    Type = 12
	TypeQ5_K    Type = 13
	TypeQ6_K    Type = 14
	TypeQ8_K    Type = 15
	TypeIQ2_XXS Type = 16
	TypeIQ2_XS  Type = 17
	TypeIQ3_XXS Type = 18
	TypeIQ1_S   Type = 19
	TypeIQ4_NL  Type = 20
	TypeIQ3_S   Type = 21
	TypeIQ2_S   Type = 22
	TypeIQ4_XS  Type = 23
	TypeI8      Type = 24
	TypeI16     Type = 25
	TypeI32     Type = 26
	TypeI64     Type = 27
	TypeF64     Type = 28
	TypeIQ1_M   Type = 29
	TypeBF16    Type = 30
	// TypeQ4_0_4_4 Type = 31 support has been removed from gguf files
	// TypeQ4_0_4_8 Type = 32
	// TypeQ4_0_8_8 Type = 33
	TypeTQ1_0 Type = 34
	TypeTQ2_0 Type = 35
	// TypeIQ4_NL_4_4 Type = 36
	// TypeIQ4_NL_4_8 Type = 37
	// TypeIQ4_NL_8_8 Type = 38
	TypeCOUNT Type = 39
)

// TensorFlag represents flags for tensors
type TensorFlag int

const (
	TensorFlagInput  TensorFlag = 1
	TensorFlagOutput TensorFlag = 2
	TensorFlagParam  TensorFlag = 4
	TensorFlagLoss   TensorFlag = 8
)

const (
	DefaultGraphSize int = 2048
)

// InitParams represents initialization parameters for a context
type InitParams struct {
	MemSize   int
	MemBuffer unsafe.Pointer
	NoAlloc   bool
}

// Context represents a ggml context
type Context struct {
	ptr *C.struct_ggml_context
}

// Ptr returns the underlying C pointer
func (c *Context) Ptr() unsafe.Pointer {
	return unsafe.Pointer(c.ptr)
}

func TensorOverhead() int {
	return int(C.ggml_tensor_overhead())
}

func GraphOverhead() int {
	return int(C.ggml_graph_overhead())
}
