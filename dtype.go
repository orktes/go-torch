package torch

// #include "torch.hpp"
import "C"
import "reflect"

// DType tensor scalar data type
type DType C.Torch_DataType

const (
	// Byte byte tensors (go type uint8)
	Byte DType = C.Torch_Byte
	// Char char tensor (go type int8)
	Char DType = C.Torch_Char
	// Int int tensor (go type int32)
	Int DType = C.Torch_Int
	// Long long tensor (go type int64)
	Long DType = C.Torch_Long
	// Float tensor (go type float32)
	Float DType = C.Torch_Float
	// Double tensor  (go type float64)
	Double DType = C.Torch_Double
)

var types = []struct {
	typ      reflect.Type
	dataType C.Torch_DataType
}{
	{reflect.TypeOf(uint8(0)), C.Torch_Byte},
	{reflect.TypeOf(int8(0)), C.Torch_Char},
	// {reflect.TypeOf(int16(0)), C.Torch_Short},
	{reflect.TypeOf(int32(0)), C.Torch_Int},
	{reflect.TypeOf(int64(0)), C.Torch_Long},
	// {reflect.TypeOf(float16(0)), C.Torch_Half},
	{reflect.TypeOf(float32(0)), C.Torch_Float},
	{reflect.TypeOf(float64(0)), C.Torch_Double},
}
