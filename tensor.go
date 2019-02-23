package torch

// #include "torch.hpp"
// #include <stdlib.h>
import "C"
import (
	"bytes"
	"encoding/binary"
	"fmt"
	"reflect"
	"runtime"
	"unsafe"
)

// Tensor holds a multi-dimensional array of elements of a single data type.
type Tensor struct {
	context C.Torch_TensorContext
	goData  unsafe.Pointer
}

// NewTensor converts from a Go value to a Tensor. Valid values are scalars, slices, and arrays. Every element of a slice must have the same length so that the resulting Tensor has a valid shape.
func NewTensor(value interface{}) (*Tensor, error) {
	val := reflect.ValueOf(value)
	shape, dataType, err := shapeAndDataTypeOf(val)
	if err != nil {
		return nil, err
	}
	return NewTensorWithShape(value, shape, dataType)
}

// NewTensorWithShape converts a single dimensional Go array or slice into a Tensor with given shape
func NewTensorWithShape(value interface{}, shape []int64, dt DType) (*Tensor, error) {
	nflattened := numElements(shape)
	nbytes := typeOf(dt, nil).Size() * uintptr(nflattened)
	dataPtr := C.malloc(C.size_t(nbytes))
	dataSlice := (*[1 << 30]byte)(dataPtr)[:nbytes:nbytes]

	buf := bytes.NewBuffer(dataSlice[:0:nbytes])
	encodeTensor(buf, reflect.ValueOf(value), shape)

	ctx := createTensor(dataPtr, shape, dt)
	t := tensorWithContext(ctx)
	t.goData = dataPtr

	return t, nil
}

func tensorWithContext(ctx C.Torch_TensorContext) *Tensor {
	t := &Tensor{
		context: ctx,
	}

	runtime.SetFinalizer(t, (*Tensor).finalize)

	return t
}

// DType returns tensors datatype
func (t *Tensor) DType() DType {
	return DType(C.Torch_TensorType(t.context))
}

// Value returns tensors value as a go type
func (t *Tensor) Value() interface{} {
	dt := t.DType()
	shape := t.Shape()

	typ := typeOf(dt, shape)
	val := reflect.New(typ)

	nflattened := numElements(shape)
	nbytes := typeOf(dt, nil).Size() * uintptr(nflattened)

	dataPtr := C.Torch_TensorValue(t.context)
	dataSlice := (*[1 << 30]byte)(dataPtr)[:nbytes:nbytes]

	if err := decodeTensor(bytes.NewReader(dataSlice), shape, typ, val); err != nil {
		panic(fmt.Sprintf("unable to decode Tensor of type %v and shape %v - %v", dt, shape, err))
	}

	return reflect.Indirect(val).Interface()
}

// Shape returns tensors shape
func (t *Tensor) Shape() []int64 {
	var size C.ulong
	shape := C.Torch_TensorShape(t.context, &size)
	slice := (*[1 << 30]int64)(unsafe.Pointer(shape))[:size:size]
	return slice
}

func (t *Tensor) finalize() {
	C.Torch_DeleteTensor(t.context)
	if t.goData != nil {
		C.free(t.goData)
	}
}

func createTensor(ptr unsafe.Pointer, shape []int64, dtype DType) C.Torch_TensorContext {
	var shapePtr *C.int64_t
	if len(shape) > 0 {
		shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}
	ctx := C.Torch_NewTensor(ptr, shapePtr, C.int(len(shape)), C.Torch_DataType(dtype))

	runtime.KeepAlive(shape)
	runtime.KeepAlive(ptr)

	return ctx
}

// shapeAndDataTypeOf returns the data type and shape of the Tensor
// corresponding to a Go type.
func shapeAndDataTypeOf(val reflect.Value) (shape []int64, dt DType, err error) {
	typ := val.Type()
	for typ.Kind() == reflect.Array || typ.Kind() == reflect.Slice {
		shape = append(shape, int64(val.Len()))
		if val.Len() > 0 {
			// In order to check tensor structure properly in general case we need to iterate over all slices of the tensor to check sizes match
			// Since we already going to iterate over all elements in encodeTensor() let's
			// 1) do the actual check in encodeTensor() to save some cpu cycles here
			// 2) assume the shape is represented by lengths of elements with zero index in each dimension
			val = val.Index(0)
		}
		typ = typ.Elem()
	}
	for _, t := range types {
		if typ.Kind() == t.typ.Kind() {
			return shape, DType(t.dataType), nil
		}
	}
	return shape, dt, fmt.Errorf("unsupported type %v", typ)
}

// decodeTensor decodes the Tensor from the buffer to ptr using the format
// specified in c_api.h. Use stringDecoder for String tensors.
func decodeTensor(r *bytes.Reader, shape []int64, typ reflect.Type, ptr reflect.Value) error {
	switch typ.Kind() {
	case reflect.Bool:
		b, err := r.ReadByte()
		if err != nil {
			return err
		}
		ptr.Elem().SetBool(b == 1)
	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
		if err := binary.Read(r, nativeEndian, ptr.Interface()); err != nil {
			return err
		}

	case reflect.Slice:
		val := reflect.Indirect(ptr)
		val.Set(reflect.MakeSlice(typ, int(shape[0]), int(shape[0])))

		// Optimization: if only one dimension is left we can use binary.Read() directly for this slice
		if len(shape) == 1 && val.Len() > 0 {
			switch val.Index(0).Kind() {
			case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
				return binary.Read(r, nativeEndian, val.Interface())
			}
		}

		for i := 0; i < val.Len(); i++ {
			if err := decodeTensor(r, shape[1:], typ.Elem(), val.Index(i).Addr()); err != nil {
				return err
			}
		}

	default:
		return fmt.Errorf("unsupported type %v", typ)
	}
	return nil
}

func encodeTensor(w *bytes.Buffer, v reflect.Value, shape []int64) error {
	switch v.Kind() {
	case reflect.Bool:
		b := byte(0)
		if v.Bool() {
			b = 1
		}
		if err := w.WriteByte(b); err != nil {
			return err
		}
	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
		if err := binary.Write(w, nativeEndian, v.Interface()); err != nil {
			return err
		}

	case reflect.Array, reflect.Slice:
		// If current dimension is a slice, verify that it has the expected size
		// Go's type system makes that guarantee for arrays.
		if v.Kind() == reflect.Slice {
			expected := int(shape[0])
			if v.Len() != expected {
				return fmt.Errorf("mismatched slice lengths: %d and %d", v.Len(), expected)
			}
		}

		// Optimisation: if only one dimension is left we can use binary.Write() directly for this slice
		if len(shape) == 1 && v.Len() > 0 {
			switch v.Index(0).Kind() {
			case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
				return binary.Write(w, nativeEndian, v.Interface())
			}
		}

		subShape := shape[1:]
		for i := 0; i < v.Len(); i++ {
			err := encodeTensor(w, v.Index(i), subShape)
			if err != nil {
				return err
			}
		}

	default:
		return fmt.Errorf("unsupported type %v", v.Type())
	}
	return nil
}

// typeOf converts from a DType and Shape to the equivalent Go type.
func typeOf(dt DType, shape []int64) reflect.Type {
	var ret reflect.Type
	for _, t := range types {
		if dt == DType(t.dataType) {
			ret = t.typ
			break
		}
	}
	if ret == nil {
		// TODO get tensor name
		panic(fmt.Sprintf("Unsupported DType %d", int(dt)))
	}
	for range shape {
		ret = reflect.SliceOf(ret)
	}
	return ret
}

func numElements(shape []int64) int64 {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	return n
}

var nativeEndian binary.ByteOrder

func init() {
	buf := [2]byte{}
	*(*uint16)(unsafe.Pointer(&buf[0])) = uint16(0xABCD)

	switch buf {
	case [2]byte{0xCD, 0xAB}:
		nativeEndian = binary.LittleEndian
	case [2]byte{0xAB, 0xCD}:
		nativeEndian = binary.BigEndian
	default:
		panic("Could not determine native endianness.")
	}
}

// PrintTensors prints tensors contents
func PrintTensors(inputs ...*Tensor) {
	contexts := make([]C.Torch_TensorContext, len(inputs))
	for i, t := range inputs {
		contexts[i] = t.context
	}

	C.Torch_PrintTensors((*C.Torch_TensorContext)(&contexts[0]), C.ulong(len(contexts)))

	runtime.KeepAlive(inputs)

}
