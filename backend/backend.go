package backend

/*
#cgo CFLAGS: -I${SRCDIR}/../ggml/include
#cgo LDFLAGS: -L${SRCDIR}/../ggml/build/src -lggml -lggml-base -lggml-cpu -lc++
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
*/
import "C"
import (
	"unsafe"

	"github.com/rengensheng/ggml-go/ctx"
	"github.com/rengensheng/ggml-go/lm"
	"github.com/rengensheng/ggml-go/tensor"
)

// Backend represents a ggml backend
type Backend struct {
	ptr C.ggml_backend_t
}

// BufferType represents a ggml backend buffer type
type BufferType struct {
	ptr C.ggml_backend_buffer_type_t
}

// Buffer represents a ggml backend buffer
type Buffer struct {
	ptr C.ggml_backend_buffer_t
}

// Graph represents a computation graph
type Graph struct {
	ptr *C.struct_ggml_cgraph
}

// NewCPUBackend creates a new CPU backend
func NewCPUBackend() *Backend {
	cBackend := C.ggml_backend_cpu_init()
	return &Backend{ptr: cBackend}
}

// Free frees the backend
func (b *Backend) Free() {
	C.ggml_backend_free(b.ptr)
}

// Name returns the name of the backend
func (b *Backend) Name() string {
	cName := C.ggml_backend_name(b.ptr)
	return C.GoString(cName)
}

// GetDefaultBufferType returns the default buffer type for the backend
func (b *Backend) GetDefaultBufferType() *BufferType {
	cBuft := C.ggml_backend_get_default_buffer_type(b.ptr)
	return &BufferType{ptr: cBuft}
}

// AllocBuffer allocates a buffer of the specified size
func (b *Backend) AllocBuffer(size int) *Buffer {
	cBuffer := C.ggml_backend_alloc_buffer(b.ptr, C.size_t(size))
	return &Buffer{ptr: cBuffer}
}

// GetAlignment returns the alignment requirement for the backend
func (b *Backend) GetAlignment() int {
	return int(C.ggml_backend_get_alignment(b.ptr))
}

// TensorSetAsync asynchronously sets tensor data
func (b *Backend) TensorSetAsync(t *tensor.Tensor, data unsafe.Pointer, offset, size int) {
	C.ggml_backend_tensor_set_async(b.ptr, (*C.struct_ggml_tensor)(t.Ptr()), data, C.size_t(offset), C.size_t(size))
}

// TensorGetAsync asynchronously gets tensor data
func (b *Backend) TensorGetAsync(t *tensor.Tensor, data unsafe.Pointer, offset, size int) {
	C.ggml_backend_tensor_get_async(b.ptr, (*C.struct_ggml_tensor)(t.Ptr()), data, C.size_t(offset), C.size_t(size))
}

// TensorSet sets tensor data
func (b *Backend) TensorSet(t *tensor.Tensor, data unsafe.Pointer, offset, size int) {
	C.ggml_backend_tensor_set((*C.struct_ggml_tensor)(t.Ptr()), data, C.size_t(offset), C.size_t(size))
}

// TensorGet gets tensor data
func (b *Backend) TensorGet(t *tensor.Tensor, data unsafe.Pointer, offset, size int) {
	C.ggml_backend_tensor_get((*C.struct_ggml_tensor)(t.Ptr()), data, C.size_t(offset), C.size_t(size))
}

// Synchronize synchronizes the backend
func (b *Backend) Synchronize() {
	C.ggml_backend_synchronize(b.ptr)
}

// GraphCompute computes a graph
func (b *Backend) GraphCompute(graph *Graph) lm.Status {
	cStatus := C.ggml_backend_graph_compute(b.ptr, graph.ptr)
	return lm.Status(cStatus)
}

// GraphComputeAsync computes a graph asynchronously
func (b *Backend) GraphComputeAsync(graph *Graph) lm.Status {
	cStatus := C.ggml_backend_graph_compute_async(b.ptr, graph.ptr)
	return lm.Status(cStatus)
}

// NewGraph creates a new computation graph
func NewGraph(ctx *ctx.Context) *Graph {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cGraph := C.ggml_new_graph(cCtx)
	return &Graph{ptr: cGraph}
}

// BuildForwardExpand builds the forward graph
func (g *Graph) BuildForwardExpand(tensor *tensor.Tensor) {
	C.ggml_build_forward_expand(g.ptr, (*C.struct_ggml_tensor)(tensor.Ptr()))
}

// Ptr returns the underlying C pointer
func (g *Graph) Ptr() unsafe.Pointer {
	return unsafe.Pointer(g.ptr)
}

// BufferType methods

// Name returns the name of the buffer type
func (bt *BufferType) Name() string {
	cName := C.ggml_backend_buft_name(bt.ptr)
	return C.GoString(cName)
}

// AllocBuffer allocates a buffer of the specified size
func (bt *BufferType) AllocBuffer(size int) *Buffer {
	cBuffer := C.ggml_backend_buft_alloc_buffer(bt.ptr, C.size_t(size))
	return &Buffer{ptr: cBuffer}
}

// GetAlignment returns the alignment requirement
func (bt *BufferType) GetAlignment() int {
	return int(C.ggml_backend_buft_get_alignment(bt.ptr))
}

// GetMaxSize returns the maximum buffer size
func (bt *BufferType) GetMaxSize() int {
	return int(C.ggml_backend_buft_get_max_size(bt.ptr))
}

// IsHost checks if this is a host buffer type
func (bt *BufferType) IsHost() bool {
	return bool(C.ggml_backend_buft_is_host(bt.ptr))
}

// Buffer methods

// Name returns the name of the buffer
func (b *Buffer) Name() string {
	cName := C.ggml_backend_buffer_name(b.ptr)
	return C.GoString(cName)
}

// Free frees the buffer
func (b *Buffer) Free() {
	C.ggml_backend_buffer_free(b.ptr)
}

// GetBase returns the base pointer of the buffer
func (b *Buffer) GetBase() unsafe.Pointer {
	return C.ggml_backend_buffer_get_base(b.ptr)
}

// GetSize returns the size of the buffer
func (b *Buffer) GetSize() int {
	return int(C.ggml_backend_buffer_get_size(b.ptr))
}

// GetAlignment returns the alignment requirement
func (b *Buffer) GetAlignment() int {
	return int(C.ggml_backend_buffer_get_alignment(b.ptr))
}

// Clear clears the buffer with the specified value
func (b *Buffer) Clear(value byte) {
	C.ggml_backend_buffer_clear(b.ptr, C.uint8_t(value))
}

// IsHost checks if this is a host buffer
func (b *Buffer) IsHost() bool {
	return bool(C.ggml_backend_buffer_is_host(b.ptr))
}
