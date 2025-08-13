package tensor

import (
	"testing"
	"unsafe"

	"github.com/rengensheng/ggml-go/ctx"
	"github.com/rengensheng/ggml-go/lm"
)

func TestSoftMax(t *testing.T) {
	// Initialize context
	params := lm.InitParams{
		MemSize: 1024 * 1024,
	}
	c := ctx.New(params)
	defer c.Free()

	// Create a 1D tensor with some test data
	tensor := NewTensor1D(c, lm.TypeF32, 4)

	// Set some values: [1.0, 2.0, 3.0, 4.0]
	tensor.SetF32(0, 1.0)
	tensor.SetF32(1, 2.0)
	tensor.SetF32(2, 3.0)
	tensor.SetF32(3, 4.0)

	// Apply softmax
	result := SoftMax(c, tensor)

	// For now, just verify that the operation doesn't crash and returns a valid tensor
	if result == nil {
		t.Error("SoftMax returned nil")
	}

	if result.Ptr() == unsafe.Pointer(nil) {
		t.Error("SoftMax returned tensor with nil pointer")
	}
}

func TestSoftMaxInplace(t *testing.T) {
	// Initialize context
	params := lm.InitParams{
		MemSize: 1024 * 1024,
	}
	c := ctx.New(params)
	defer c.Free()

	// Create a 1D tensor with some test data
	tensor := NewTensor1D(c, lm.TypeF32, 4)

	// Set some values: [1.0, 2.0, 3.0, 4.0]
	tensor.SetF32(0, 1.0)
	tensor.SetF32(1, 2.0)
	tensor.SetF32(2, 3.0)
	tensor.SetF32(3, 4.0)

	// Apply softmax in-place
	result := SoftMaxInplace(c, tensor)

	// For now, just verify that the operation doesn't crash and returns a valid tensor
	if result == nil {
		t.Error("SoftMaxInplace returned nil")
	}

	if result.Ptr() == unsafe.Pointer(nil) {
		t.Error("SoftMaxInplace returned tensor with nil pointer")
	}
}

func TestSoftMaxExt(t *testing.T) {
	// Initialize context
	params := lm.InitParams{
		MemSize: 1024 * 1024,
	}
	c := ctx.New(params)
	defer c.Free()

	// Create a 1D tensor with some test data
	tensor := NewTensor1D(c, lm.TypeF32, 4)

	// Set some values: [1.0, 2.0, 3.0, 4.0]
	tensor.SetF32(0, 1.0)
	tensor.SetF32(1, 2.0)
	tensor.SetF32(2, 3.0)
	tensor.SetF32(3, 4.0)

	// Create a mask tensor
	mask := NewTensor1D(c, lm.TypeF32, 4)
	mask.SetF32(0, 1.0)
	mask.SetF32(1, 1.0)
	mask.SetF32(2, 1.0)
	mask.SetF32(3, 1.0)

	// Apply extended softmax
	result := SoftMaxExt(c, tensor, mask, 1.0, 0.0)

	// For now, just verify that the operation doesn't crash and returns a valid tensor
	if result == nil {
		t.Error("SoftMaxExt returned nil")
	}

	if result.Ptr() == unsafe.Pointer(nil) {
		t.Error("SoftMaxExt returned tensor with nil pointer")
	}
}

func TestSoftMaxExtBack(t *testing.T) {
	// Initialize context
	params := lm.InitParams{
		MemSize: 1024 * 1024,
	}
	c := ctx.New(params)
	defer c.Free()

	// Create a 1D tensor with some test data
	tensorA := NewTensor1D(c, lm.TypeF32, 4)

	// Set some values: [1.0, 2.0, 3.0, 4.0]
	tensorA.SetF32(0, 1.0)
	tensorA.SetF32(1, 2.0)
	tensorA.SetF32(2, 3.0)
	tensorA.SetF32(3, 4.0)

	// Create another tensor
	tensorB := NewTensor1D(c, lm.TypeF32, 4)
	tensorB.SetF32(0, 0.1)
	tensorB.SetF32(1, 0.2)
	tensorB.SetF32(2, 0.3)
	tensorB.SetF32(3, 0.4)

	// Apply softmax backward
	result := SoftMaxExtBack(c, tensorA, tensorB, 1.0, 0.0)

	// For now, just verify that the operation doesn't crash and returns a valid tensor
	if result == nil {
		t.Error("SoftMaxExtBack returned nil")
	}

	if result.Ptr() == unsafe.Pointer(nil) {
		t.Error("SoftMaxExtBack returned tensor with nil pointer")
	}
}

func TestSoftMaxExtBackInplace(t *testing.T) {
	// Initialize context
	params := lm.InitParams{
		MemSize: 1024 * 1024,
	}
	c := ctx.New(params)
	defer c.Free()

	// Create a 1D tensor with some test data
	tensorA := NewTensor1D(c, lm.TypeF32, 4)

	// Set some values: [1.0, 2.0, 3.0, 4.0]
	tensorA.SetF32(0, 1.0)
	tensorA.SetF32(1, 2.0)
	tensorA.SetF32(2, 3.0)
	tensorA.SetF32(3, 4.0)

	// Create another tensor
	tensorB := NewTensor1D(c, lm.TypeF32, 4)
	tensorB.SetF32(0, 0.1)
	tensorB.SetF32(1, 0.2)
	tensorB.SetF32(2, 0.3)
	tensorB.SetF32(3, 0.4)

	// Apply softmax backward inplace
	result := SoftMaxExtBackInplace(c, tensorA, tensorB, 1.0, 0.0)

	// For now, just verify that the operation doesn't crash and returns a valid tensor
	if result == nil {
		t.Error("SoftMaxExtBackInplace returned nil")
	}

	if result.Ptr() == unsafe.Pointer(nil) {
		t.Error("SoftMaxExtBackInplace returned tensor with nil pointer")
	}
}

func TestActivationFunctions(t *testing.T) {
	// Initialize context
	params := lm.InitParams{
		MemSize: 1024 * 1024,
	}
	c := ctx.New(params)
	defer c.Free()

	// Create a 1D tensor with some test data
	tensor := NewTensor1D(c, lm.TypeF32, 4)
	tensor.SetF32(0, -1.0)
	tensor.SetF32(1, 0.0)
	tensor.SetF32(2, 1.0)
	tensor.SetF32(3, 2.0)

	// Test activation functions
	activationFuncs := []func(*ctx.Context, *Tensor) *Tensor{
		Neg, Sin, Cos, Tanh, Sigmoid, GELU, QuickGELU, SILU, ReLU,
	}

	for _, fn := range activationFuncs {
		result := fn(c, tensor)
		if result == nil {
			t.Error("Activation function returned nil")
		}
		if result.Ptr() == unsafe.Pointer(nil) {
			t.Error("Activation function returned tensor with nil pointer")
		}
	}
}

func TestMathOperations(t *testing.T) {
	// Initialize context
	params := lm.InitParams{
		MemSize: 1024 * 1024,
	}
	c := ctx.New(params)
	defer c.Free()

	// Create tensors with test data
	tensorA := NewTensor1D(c, lm.TypeF32, 4)
	tensorA.SetF32(0, 1.0)
	tensorA.SetF32(1, 2.0)
	tensorA.SetF32(2, 3.0)
	tensorA.SetF32(3, 4.0)

	tensorB := NewTensor1D(c, lm.TypeF32, 4)
	tensorB.SetF32(0, 0.5)
	tensorB.SetF32(1, 1.5)
	tensorB.SetF32(2, 2.5)
	tensorB.SetF32(3, 3.5)

	// Test math operations
	mathFuncs := []func(*ctx.Context, *Tensor, *Tensor) *Tensor{
		Add, Sub, Mul, Div,
	}

	for _, fn := range mathFuncs {
		result := fn(c, tensorA, tensorB)
		if result == nil {
			t.Error("Math operation returned nil")
		}
		if result.Ptr() == unsafe.Pointer(nil) {
			t.Error("Math operation returned tensor with nil pointer")
		}
	}
}

func TestNormalization(t *testing.T) {
	// Initialize context
	params := lm.InitParams{
		MemSize: 1024 * 1024,
	}
	c := ctx.New(params)
	defer c.Free()

	// Create a tensor with test data
	tensor := NewTensor1D(c, lm.TypeF32, 4)
	tensor.SetF32(0, 1.0)
	tensor.SetF32(1, 2.0)
	tensor.SetF32(2, 3.0)
	tensor.SetF32(3, 4.0)

	// Test normalization functions
	result := RMSNorm(c, tensor, 1e-5)
	if result == nil {
		t.Error("RMSNorm returned nil")
	}
	if result.Ptr() == unsafe.Pointer(nil) {
		t.Error("RMSNorm returned tensor with nil pointer")
	}
}

func TestUtilityFunctions(t *testing.T) {
	// Initialize context
	params := lm.InitParams{
		MemSize: 1024 * 1024,
	}
	c := ctx.New(params)
	defer c.Free()

	// Create a tensor with test data
	tensor := NewTensor1D(c, lm.TypeF32, 4)
	tensor.SetF32(0, 1.0)
	tensor.SetF32(1, 2.0)
	tensor.SetF32(2, 3.0)
	tensor.SetF32(3, 4.0)

	// Test utility functions
	result := SumRows(c, tensor)
	if result == nil {
		t.Error("SumRows returned nil")
	}
	if result.Ptr() == unsafe.Pointer(nil) {
		t.Error("SumRows returned tensor with nil pointer")
	}

	result = Duplicate(c, tensor)
	if result == nil {
		t.Error("Duplicate returned nil")
	}
	if result.Ptr() == unsafe.Pointer(nil) {
		t.Error("Duplicate returned tensor with nil pointer")
	}
}
