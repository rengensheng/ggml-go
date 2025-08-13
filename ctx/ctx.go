package ctx

/*
#cgo CFLAGS: -I${SRCDIR}/../ggml/include
#cgo LDFLAGS: -L${SRCDIR}/../ggml/build/src -lggml -lggml-base -lggml-cpu -lc++
#include "ggml.h"
#include "ggml-cpu.h"
*/
import "C"
import (
	"unsafe"

	"github.com/rengensheng/ggml-go/lm"
)

// Context represents a ggml context
type Context struct {
	ptr *C.struct_ggml_context
}

// New creates a new context with the given parameters
func New(params lm.InitParams) *Context {
	cParams := C.struct_ggml_init_params{
		mem_size:   C.size_t(params.MemSize),
		mem_buffer: params.MemBuffer,
		no_alloc:   C.bool(params.NoAlloc),
	}
	cCtx := C.ggml_init(cParams)
	return &Context{ptr: cCtx}
}

// Free frees the context
func (c *Context) Free() {
	C.ggml_free(c.ptr)
}

// Ptr returns the underlying C pointer
func (c *Context) Ptr() unsafe.Pointer {
	return unsafe.Pointer(c.ptr)
}

// UsedMem returns the amount of memory used by the context
func (c *Context) UsedMem() int {
	return int(C.ggml_used_mem(c.ptr))
}

// GetMemBuffer returns the memory buffer of the context
func (c *Context) GetMemBuffer() unsafe.Pointer {
	return C.ggml_get_mem_buffer(c.ptr)
}

// GetMemSize returns the size of the memory buffer
func (c *Context) GetMemSize() int {
	return int(C.ggml_get_mem_size(c.ptr))
}

// Reset resets the context
func (c *Context) Reset() {
	C.ggml_reset(c.ptr)
}
