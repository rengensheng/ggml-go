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
	TypeQ5_0 Type = 6
	TypeQ5_1 Type = 7
	TypeQ8_0 Type = 8
	TypeQ8_1 Type = 9
)

// TensorFlag represents flags for tensors
type TensorFlag int

const (
	TensorFlagInput  TensorFlag = 1
	TensorFlagOutput TensorFlag = 2
	TensorFlagParam  TensorFlag = 4
	TensorFlagLoss   TensorFlag = 8
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
