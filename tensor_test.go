package torch

import (
	"testing"
	"unsafe"
)

func Test_createTensor(t *testing.T) {
	data := []float32{1, 2}
	ctx := createTensor(unsafe.Pointer(&data[0]), []int64{2}, Float)
	if ctx == nil {
		t.Error("should have returned an array")
	}
}

func Test_NewTensor(t *testing.T) {
	tensor, err := NewTensor([]float32{1, 2})
	if err != nil {
		t.Error(err)
	}
	if tensor == nil {
		t.Error("no tensor returned")
	}

	if tensor.DType() != Float {
		t.Error("should be a float tensor")
	}

	val := tensor.Value().([]float32)
	if val[0] != 1.0 {
		t.Error("wrong value returned by tensor")
	}
	if val[1] != 2.0 {
		t.Error("wrong value returned by tensor")
	}

	if tensor.Shape()[0] != 2 {
		t.Error("wrong shape returned by tensor")
	}
}

func Test_PrintTensors(t *testing.T) {
	a, _ := NewTensor([]float32{1, 2})
	b, _ := NewTensor([]float32{1, 2})

	PrintTensors(a, b)

}

func Benchmark_NewTensor(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NewTensor([]float32{1, 2})
	}
}
