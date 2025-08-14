package utils

import "math"

// Softmax 对 x 做 in-place 的 softmax；若 dst 为 nil 则返回新切片
func Softmax(x []float32) []float32 {
	if len(x) == 0 {
		return x
	}
	dst := make([]float32, len(x))

	// 1. 减去最大值防溢出
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	// 2. exp & 求和
	var sum float32 = 0.0
	for i, v := range x {
		dst[i] = float32(math.Exp(float64(v - maxVal)))
		sum += dst[i]
	}

	// 3. 归一化
	for i := range dst {
		dst[i] /= sum
	}
	return dst
}

// ArgMax 返回最大值所在下标；若切片为空返回 -1
func ArgMax(x []float32) int {
	if len(x) == 0 {
		return -1
	}
	idx := 0
	maxVal := x[0]
	for i, v := range x[1:] {
		if v > maxVal {
			maxVal = v
			idx = i + 1
		}
	}
	return idx
}
