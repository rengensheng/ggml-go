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
	"fmt"
	"io"
	"os"
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

func NewTensorI32(ctx *ctx.Context, val int32) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_new_i32(cCtx, C.int32_t(val))
	return &Tensor{ptr: cTensor}
}

func NewTensorF32(ctx *ctx.Context, val float32) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_new_f32(cCtx, C.float(val))
	return &Tensor{ptr: cTensor}
}

func (t *Tensor) GetNe(index int) int32 {
	return int32(t.ptr.ne[index])
}

func (t *Tensor) NElements() int {
	return int(C.ggml_nelements(t.ptr))
}

func (t *Tensor) NBytes() int {
	return int(C.ggml_nbytes(t.ptr))
}

func (t *Tensor) DataPtr() unsafe.Pointer {
	return t.ptr.data
}

func (t *Tensor) ReadTensorData(f *os.File) error {
	dataPtr := t.DataPtr()
	nBytes := t.NBytes()
	fmt.Println("读取的字节数", nBytes)
	buf := unsafe.Slice((*byte)(dataPtr), nBytes)
	_, err := io.ReadFull(f, buf)
	return err
}

func (t *Tensor) ToString() string {
	nBytes := t.NBytes()
	buf := unsafe.Slice((*byte)(t.DataPtr()), nBytes)
	return string(buf)
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

func (t *Tensor) Shape() []int64 {
	dims := make([]int64, 4)
	for i := 0; i < 4; i++ {
		dims[i] = int64(t.ptr.ne[i]) // 或者 int32，看你要的精度
	}
	return dims
}

func (t *Tensor) Empty() bool {
	return bool(C.ggml_is_empty(t.ptr))
}

// Add performs element-wise addition of two tensors
func Add(ctx *ctx.Context, a, b *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_add(cCtx, a.ptr, b.ptr)
	return &Tensor{ptr: cTensor}
}

func Add1(ctx *ctx.Context, a, b *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_add1(cCtx, a.ptr, b.ptr)
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

// Neg computes the negation of a tensor
func Neg(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_neg(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// NegInplace computes the negation of a tensor in-place
func NegInplace(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_neg_inplace(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// MulMat performs matrix multiplication
func MulMat(ctx *ctx.Context, a, b *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_mul_mat(cCtx, a.ptr, b.ptr)
	return &Tensor{ptr: cTensor}
}

// MulMatID performs indexed matrix multiplication
func MulMatID(ctx *ctx.Context, as, b, ids *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_mul_mat_id(cCtx, as.ptr, b.ptr, ids.ptr)
	return &Tensor{ptr: cTensor}
}

// Scale scales a tensor by a scalar value
func Scale(ctx *ctx.Context, a *Tensor, s float64) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cS := C.float(s)
	cTensor := C.ggml_scale(cCtx, a.ptr, cS)
	return &Tensor{ptr: cTensor}
}

// ScaleInplace scales a tensor by a scalar value in-place
func ScaleInplace(ctx *ctx.Context, a *Tensor, s float64) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cS := C.float(s)
	cTensor := C.ggml_scale_inplace(cCtx, a.ptr, cS)
	return &Tensor{ptr: cTensor}
}

// LayerNorm performs layer normalization
func LayerNorm(ctx *ctx.Context, a *Tensor, eps float32) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cEps := C.float(eps)
	cTensor := C.ggml_norm(cCtx, a.ptr, cEps)
	return &Tensor{ptr: cTensor}
}

// LayerNormInplace performs layer normalization in-place
func LayerNormInplace(ctx *ctx.Context, a *Tensor, eps float32) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cEps := C.float(eps)
	cTensor := C.ggml_norm_inplace(cCtx, a.ptr, cEps)
	return &Tensor{ptr: cTensor}
}

// RMSNorm performs RMS normalization
func RMSNorm(ctx *ctx.Context, a *Tensor, eps float32) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cEps := C.float(eps)
	cTensor := C.ggml_rms_norm(cCtx, a.ptr, cEps)
	return &Tensor{ptr: cTensor}
}

// RMSNormInplace performs RMS normalization in-place
func RMSNormInplace(ctx *ctx.Context, a *Tensor, eps float32) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cEps := C.float(eps)
	cTensor := C.ggml_rms_norm_inplace(cCtx, a.ptr, cEps)
	return &Tensor{ptr: cTensor}
}

// Repeat repeats a tensor along specified dimensions
func RepeatDim(ctx *ctx.Context, a *Tensor, dim, n int) *Tensor {
	// This is a simplified implementation, actual implementation would depend on specific use case
	// For now, we'll create a tensor with the repeated dimensions
	ne := make([]int64, 4)
	for i := 0; i < 4; i++ {
		ne[i] = int64(a.ptr.ne[i])
	}
	ne[dim] *= int64(n)

	// Create a tensor with the new shape for repeat operation
	// Using F32 as default type since we can't access the C struct field directly
	b := NewTensor4D(ctx, lm.TypeF32, int(ne[0]), int(ne[1]), int(ne[2]), int(ne[3]))
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_repeat(cCtx, a.ptr, b.ptr)
	return &Tensor{ptr: cTensor}
}

// Stack stacks tensors along a dimension
func Stack(ctx *ctx.Context, dim int, s ...*Tensor) *Tensor {
	if len(s) == 0 {
		return nil
	}

	// Calculate the output shape
	ne := make([]int64, 4)
	for i := 0; i < 4; i++ {
		ne[i] = int64(s[0].ptr.ne[i])
	}
	ne[dim] = int64(len(s)) * ne[dim]

	// Create output tensor
	// Using F32 as default type since we can't access the C struct field directly
	out := NewTensor4D(ctx, lm.TypeF32, int(ne[0]), int(ne[1]), int(ne[2]), int(ne[3]))

	// For now, we'll just return the first tensor as a placeholder
	// A full implementation would need to properly stack all tensors
	return out
}

// Concat concatenates two tensors along a dimension
func Concat(ctx *ctx.Context, a, b *Tensor, dim int) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cDim := C.int(dim)
	cTensor := C.ggml_concat(cCtx, a.ptr, b.ptr, cDim)
	return &Tensor{ptr: cTensor}
}

// Contiguous makes a tensor contiguous in memory
func Contiguous(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_cont(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// Contiguous1D makes a tensor contiguous in memory with 1D shape
func Contiguous1D(ctx *ctx.Context, a *Tensor, ne0 int) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cNe0 := C.int64_t(ne0)
	cTensor := C.ggml_cont_1d(cCtx, a.ptr, cNe0)
	return &Tensor{ptr: cTensor}
}

// Contiguous2D makes a tensor contiguous in memory with 2D shape
func Contiguous2D(ctx *ctx.Context, a *Tensor, ne0, ne1 int) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cNe0 := C.int64_t(ne0)
	cNe1 := C.int64_t(ne1)
	cTensor := C.ggml_cont_2d(cCtx, a.ptr, cNe0, cNe1)
	return &Tensor{ptr: cTensor}
}

// Contiguous3D makes a tensor contiguous in memory with 3D shape
func Contiguous3D(ctx *ctx.Context, a *Tensor, ne0, ne1, ne2 int) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cNe0 := C.int64_t(ne0)
	cNe1 := C.int64_t(ne1)
	cNe2 := C.int64_t(ne2)
	cTensor := C.ggml_cont_3d(cCtx, a.ptr, cNe0, cNe1, cNe2)
	return &Tensor{ptr: cTensor}
}

// Contiguous4D makes a tensor contiguous in memory with 4D shape
func Contiguous4D(ctx *ctx.Context, a *Tensor, ne0, ne1, ne2, ne3 int) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cNe0 := C.int64_t(ne0)
	cNe1 := C.int64_t(ne1)
	cNe2 := C.int64_t(ne2)
	cNe3 := C.int64_t(ne3)
	cTensor := C.ggml_cont_4d(cCtx, a.ptr, cNe0, cNe1, cNe2, cNe3)
	return &Tensor{ptr: cTensor}
}

// Pad pads a tensor with zeros
func Pad(ctx *ctx.Context, a *Tensor, pad0, pad1, pad2, pad3 int) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cPad0 := C.int(pad0)
	cPad1 := C.int(pad1)
	cPad2 := C.int(pad2)
	cPad3 := C.int(pad3)
	cTensor := C.ggml_pad(cCtx, a.ptr, cPad0, cPad1, cPad2, cPad3)
	return &Tensor{ptr: cTensor}
}

// View creates a view of a tensor with specified offset and strides
func View(ctx *ctx.Context, a *Tensor, offset int, strides ...int) *Tensor {
	// Implementation depends on specific use case
	// This is a simplified placeholder implementation
	return a
}

// Copy copies a tensor
func Copy(ctx *ctx.Context, a, b *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_cpy(cCtx, a.ptr, b.ptr)
	return &Tensor{ptr: cTensor}
}

// Set sets values in a tensor
func Set(ctx *ctx.Context, a, b *Tensor, offset int, strides ...int) *Tensor {
	// This is a simplified implementation
	// A full implementation would need to handle strides properly
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cOffset := C.size_t(offset)

	// Use default strides for now
	cNb1 := C.size_t(b.ptr.nb[1])
	cNb2 := C.size_t(b.ptr.nb[2])
	cNb3 := C.size_t(b.ptr.nb[3])

	cTensor := C.ggml_set(cCtx, a.ptr, b.ptr, cNb1, cNb2, cNb3, cOffset)
	return &Tensor{ptr: cTensor}
}

// Rows gets specific rows from a tensor
func Rows(ctx *ctx.Context, a, indices *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_get_rows(cCtx, a.ptr, indices.ptr)
	return &Tensor{ptr: cTensor}
}

// IM2Col converts image to column format for convolution
func IM2Col(ctx *ctx.Context, im *Tensor, s0, s1, p0, p1, d0, d1 int) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cS0 := C.int(s0)
	cS1 := C.int(s1)
	cP0 := C.int(p0)
	cP1 := C.int(p1)
	cD0 := C.int(d0)
	cD1 := C.int(d1)
	cIs2D := C.bool(true)
	cDstType := C.enum_ggml_type(lm.TypeF32)
	cTensor := C.ggml_im2col(cCtx, im.ptr, im.ptr, cS0, cS1, cP0, cP1, cD0, cD1, cIs2D, cDstType)
	return &Tensor{ptr: cTensor}
}

// Conv2D performs 2D convolution
func Conv2D(ctx *ctx.Context, im, kernel *Tensor, s0, s1, p0, p1, d0, d1 int) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cS0 := C.int(s0)
	cS1 := C.int(s1)
	cP0 := C.int(p0)
	cP1 := C.int(p1)
	cD0 := C.int(d0)
	cD1 := C.int(d1)
	cTensor := C.ggml_conv_2d(cCtx, kernel.ptr, im.ptr, cS0, cS1, cP0, cP1, cD0, cD1)
	return &Tensor{ptr: cTensor}
}

// AvgPool2D performs 2D average pooling
func AvgPool2D(ctx *ctx.Context, a *Tensor, k, s int, p float32) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cK := C.int(k)
	cS := C.int(s)
	cP := C.float(p)
	cTensor := C.ggml_pool_2d(cCtx, a.ptr, C.GGML_OP_POOL_AVG, cK, cK, cS, cS, cP, cP)
	return &Tensor{ptr: cTensor}
}

// GELU computes the GELU activation function
func GELU(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_gelu(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// GELUInplace computes the GELU activation function in-place
func GELUInplace(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_gelu_inplace(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// QuickGELU computes the QuickGELU activation function
func QuickGELU(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_gelu_quick(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// QuickGELUInplace computes the QuickGELU activation function in-place
func QuickGELUInplace(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_gelu_quick_inplace(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// SILU computes the SILU activation function
func SILU(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_silu(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// SILUInplace computes the SILU activation function in-place
func SILUInplace(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_silu_inplace(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// ReLU computes the ReLU activation function
func ReLU(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_relu(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// ReLUInplace computes the ReLU activation function in-place
func ReLUInplace(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_relu_inplace(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// Sigmoid computes the sigmoid activation function
func Sigmoid(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_sigmoid(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// SigmoidInplace computes the sigmoid activation function in-place
func SigmoidInplace(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_sigmoid_inplace(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// Tanh computes the hyperbolic tangent activation function
func Tanh(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_tanh(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// TanhInplace computes the hyperbolic tangent activation function in-place
func TanhInplace(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_tanh_inplace(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// Sin computes the sine of a tensor
func Sin(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_sin(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// SinInplace computes the sine of a tensor in-place
func SinInplace(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_sin_inplace(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// Cos computes the cosine of a tensor
func Cos(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_cos(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// CosInplace computes the cosine of a tensor in-place
func CosInplace(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_cos_inplace(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// Clamp clamps the values of a tensor between min and max
func Clamp(ctx *ctx.Context, a *Tensor, min, max float32) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cMin := C.float(min)
	cMax := C.float(max)
	cTensor := C.ggml_clamp(cCtx, a.ptr, cMin, cMax)
	return &Tensor{ptr: cTensor}
}

// SumRows computes the sum along rows of a tensor
func SumRows(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_sum_rows(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// These functions don't exist in the C library, implementing as placeholders
// Variance computes the variance of a tensor
func Variance(ctx *ctx.Context, a *Tensor) *Tensor {
	// Placeholder implementation
	return a
}

// Stddev computes the standard deviation of a tensor
func Stddev(ctx *ctx.Context, a *Tensor) *Tensor {
	// Placeholder implementation
	return a
}

// TopK selects the top k elements per row
func TopK(ctx *ctx.Context, a *Tensor, k int) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cK := C.int(k)
	cTensor := C.ggml_top_k(cCtx, a.ptr, cK)
	return &Tensor{ptr: cTensor}
}

// Argsort sorts the elements of a tensor
func Argsort(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cOrder := C.enum_ggml_sort_order(C.GGML_SORT_ORDER_ASC)
	cTensor := C.ggml_argsort(cCtx, a.ptr, cOrder)
	return &Tensor{ptr: cTensor}
}

// Duplicate duplicates a tensor
func Duplicate(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_dup(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// DuplicateInplace duplicates a tensor in-place
func DuplicateInplace(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_dup_inplace(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// ScaledDotProductAttention computes scaled dot-product attention
func ScaledDotProductAttention(ctx *ctx.Context, q, k, v, mask *Tensor, scale float64) *Tensor {
	// This is a simplified implementation using existing functions
	// A full implementation would use the dedicated ggml function if available

	// KQ = transpose(K) * Q * scale
	kq := MulMat(ctx, Transpose(ctx, k), q)
	kq = Scale(ctx, kq, scale)

	// Apply mask if provided
	if mask != nil {
		kq = Add(ctx, kq, mask)
	}

	// Softmax
	kq = SoftMax(ctx, kq)

	// KQV = V * KQ
	kqv := MulMat(ctx, v, kq)

	return kqv
}

func SoftMax(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_soft_max(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// SetF32 sets a float32 value in a 1D tensor
func SoftMaxInplace(ctx *ctx.Context, a *Tensor) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cTensor := C.ggml_soft_max_inplace(cCtx, a.ptr)
	return &Tensor{ptr: cTensor}
}

// SoftMaxExt computes the extended softmax of a tensor with mask, scale, and max_bias
func SoftMaxExt(ctx *ctx.Context, a, mask *Tensor, scale, maxBias float32) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cScale := C.float(scale)
	cMaxBias := C.float(maxBias)
	cTensor := C.ggml_soft_max_ext(cCtx, a.ptr, mask.ptr, cScale, cMaxBias)
	return &Tensor{ptr: cTensor}
}

// SoftMaxExtBack computes the backward pass of extended softmax
func SoftMaxExtBack(ctx *ctx.Context, a, b *Tensor, scale, maxBias float32) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cScale := C.float(scale)
	cMaxBias := C.float(maxBias)
	cTensor := C.ggml_soft_max_ext_back(cCtx, a.ptr, b.ptr, cScale, cMaxBias)
	return &Tensor{ptr: cTensor}
}

// SoftMaxExtBackInplace computes the backward pass of extended softmax in-place
func SoftMaxExtBackInplace(ctx *ctx.Context, a, b *Tensor, scale, maxBias float32) *Tensor {
	cCtx := (*C.struct_ggml_context)(ctx.Ptr())
	cScale := C.float(scale)
	cMaxBias := C.float(maxBias)
	cTensor := C.ggml_soft_max_ext_back_inplace(cCtx, a.ptr, b.ptr, cScale, cMaxBias)
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

func (t *Tensor) SetZero() {
	C.ggml_set_zero(t.ptr)
}
