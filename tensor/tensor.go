package tensor

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
)

// Tensor represents a ggml tensor
type Tensor struct {
	ptr *C.struct_ggml_tensor
}

// NewTensor1D creates a new 1D tensor
func NewTensor1D(ctx *ctx.Context, typ lm.Type, ne0 int) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_new_tensor_1d(cCtx, C.enum_ggml_type(typ), C.int64_t(ne0))
	return &Tensor{ptr: cTensor}
}

// NewTensor2D creates a new 2D tensor
func NewTensor2D(ctx *ctx.Context, typ lm.Type, ne0, ne1 int) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_new_tensor_2d(cCtx, C.enum_ggml_type(typ), C.int64_t(ne0), C.int64_t(ne1))
	return &Tensor{ptr: cTensor}
}

// NewTensor3D creates a new 3D tensor
func NewTensor3D(ctx *ctx.Context, typ lm.Type, ne0, ne1, ne2 int) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_new_tensor_3d(cCtx, C.enum_ggml_type(typ), C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2))
	return &Tensor{ptr: cTensor}
}

// NewTensor4D creates a new 4D tensor
func NewTensor4D(ctx *ctx.Context, typ lm.Type, ne0, ne1, ne2, ne3 int) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_new_tensor_4d(cCtx, C.enum_ggml_type(typ), C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2), C.int64_t(ne3))
	return &Tensor{ptr: cTensor}
}

// SetParam marks a tensor as a parameter
func (t *Tensor) SetParam(ctx *ctx.Context) {
	C.ggml_set_param(t.ptr)
}

// SetFlag sets a flag on the tensor
func (t *Tensor) SetFlag(flag lm.TensorFlag) {
	t.ptr.flags |= C.int32_t(flag)
}

// GetFlag checks if a flag is set on the tensor
func (t *Tensor) GetFlag(flag lm.TensorFlag) bool {
	return (t.ptr.flags & C.int32_t(flag)) != 0
}

// Add performs element-wise addition of two tensors
func Add(ctx *ctx.Context, a, b *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_add(cCtx, a.ptr, b.ptr)
	return &Tensor{ptr: cTensor}
}

// Mul performs element-wise multiplication of two tensors
func Mul(ctx *ctx.Context, a, b *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_mul(cCtx, a.ptr, b.ptr)
	return &Tensor{ptr: cTensor}
}

// Sub performs element-wise subtraction of two tensors
func Sub(ctx *ctx.Context, a, b *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_sub(cCtx, a.ptr, b.ptr)
	return &Tensor{ptr: cTensor}
}

// Div performs element-wise division of two tensors
func Div(ctx *ctx.Context, a, b *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_div(cCtx, a.ptr, b.ptr)
	return &Tensor{ptr: cTensor}
}

// Sqr computes the square of a tensor
func Sqr(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_sqr(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// Sqrt computes the square root of a tensor
func Sqrt(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_sqrt(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// Sum computes the sum of all elements in a tensor
func Sum(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_sum(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// Mean computes the mean of all elements in a tensor
func Mean(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_mean(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// Repeat repeats a tensor
func Repeat(ctx *ctx.Context, a, b *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_repeat(cCtx, a.ptr, b.ptr)
	return &Tensor{ptr: cTensor}
}

// Reshape reshapes a tensor
func Reshape(ctx *ctx.Context, a, b *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_reshape(cCtx, a.ptr, b.ptr)
	return &Tensor{ptr: cTensor}
}

// Reshape1D reshapes a tensor to 1D
func Reshape1D(ctx *ctx.Context, a *Tensor, ne0 int) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_reshape_1d(cCtx, a.ptr, C.int64_t(ne0))
	return &Tensor{ptr: cTensor}
}

// Reshape2D reshapes a tensor to 2D
func Reshape2D(ctx *ctx.Context, a *Tensor, ne0, ne1 int) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_reshape_2d(cCtx, a.ptr, C.int64_t(ne0), C.int64_t(ne1))
	return &Tensor{ptr: cTensor}
}

// Reshape3D reshapes a tensor to 3D
func Reshape3D(ctx *ctx.Context, a *Tensor, ne0, ne1, ne2 int) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_reshape_3d(cCtx, a.ptr, C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2))
	return &Tensor{ptr: cTensor}
}

// Reshape4D reshapes a tensor to 4D
func Reshape4D(ctx *ctx.Context, a *Tensor, ne0, ne1, ne2, ne3 int) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_reshape_4d(cCtx, a.ptr, C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2), C.int64_t(ne3))
	return &Tensor{ptr: cTensor}
}

// Permute permutes the dimensions of a tensor
func Permute(ctx *ctx.Context, a *Tensor, axis0, axis1, axis2, axis3 int) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_permute(cCtx, a.ptr, C.int(axis0), C.int(axis1), C.int(axis2), C.int(axis3))
	return &Tensor{ptr: cTensor}
}

// Transpose transposes a 2D tensor
func Transpose(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_transpose(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// GetRows gets rows from a tensor
func GetRows(ctx *ctx.Context, a, b *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_get_rows(cCtx, a.ptr, b.ptr)
	return &Tensor{ptr: cTensor}
}

// SetRows sets rows in a tensor
func SetRows(ctx *ctx.Context, a, b, c *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_set_rows(cCtx, a.ptr, b.ptr, c.ptr)
	return &Tensor{ptr: cTensor}
}

// SetF32 sets a float32 value in a 1D tensor
func (t *Tensor) SetF32(index int, value float32) {
	C.ggml_set_f32_1d(t.ptr, C.int(index), C.float(value))
}

// GetF32 gets a float32 value from a 1D tensor
func (t *Tensor) GetF32(index int) float32 {
	return float32(C.ggml_get_f32_1d(t.ptr, C.int(index)))
}

// SetData sets the data of a tensor
func (t *Tensor) SetData(data unsafe.Pointer, size int) {
	C.ggml_backend_tensor_set(t.ptr, data, 0, C.size_t(size))
}

// GetData gets the data of a tensor
func (t *Tensor) GetData(data unsafe.Pointer, size int) {
	C.ggml_backend_tensor_get(t.ptr, data, 0, C.size_t(size))
}

// Ptr returns the underlying C pointer
func (t *Tensor) Ptr() unsafe.Pointer {
	return unsafe.Pointer(t.ptr)
}
