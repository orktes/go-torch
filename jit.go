package torch

// #include "torch.hpp"
// #include <stdlib.h>
//
// size_t size_of_ivalue_tuple = sizeof(Torch_IValueTuple);
// size_t size_of_ivalue = sizeof(Torch_IValue);
//
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

// JITModule is a jit compiled PyTorch module
type JITModule struct {
	context C.Torch_JITModuleContext
}

// CompileTorchScript compiles TorchScript and returns a *JITModule
func CompileTorchScript(torchScript string) (*JITModule, error) {
	cstr := C.CString(torchScript)
	defer C.free(unsafe.Pointer(cstr))

	var cErr C.Torch_Error
	ctx := C.Torch_CompileTorchScript(cstr, &cErr)
	if err := checkError(cErr); err != nil {
		return nil, err
	}

	mod := &JITModule{context: ctx}
	runtime.SetFinalizer(mod, (*JITModule).finalize)

	return mod, nil
}

// LoadJITModule loads module from file
func LoadJITModule(path string) (*JITModule, error) {
	cstr := C.CString(path)
	defer C.free(unsafe.Pointer(cstr))

	var cErr C.Torch_Error
	ctx := C.Torch_LoadJITModule(cstr, &cErr)
	if err := checkError(cErr); err != nil {
		return nil, err
	}

	mod := &JITModule{context: ctx}
	runtime.SetFinalizer(mod, (*JITModule).finalize)
	// TODO handle errors
	return mod, nil
}

// Save saves Module to given path
func (m *JITModule) Save(path string) error {
	cstr := C.CString(path)
	defer C.free(unsafe.Pointer(cstr))

	var cErr C.Torch_Error
	C.Torch_ExportJITModule(m.context, cstr, &cErr)
	if err := checkError(cErr); err != nil {
		return err
	}

	// TODO handle errors
	return nil
}

// GetMethod returns a method from a JITModule
func (m *JITModule) GetMethod(method string) (*JITModuleMethod, error) {
	cstr := C.CString(method)
	defer C.free(unsafe.Pointer(cstr))

	var cErr C.Torch_Error
	context := C.Torch_JITModuleGetMethod(m.context, cstr, &cErr)
	if err := checkError(cErr); err != nil {
		return nil, err
	}

	met := &JITModuleMethod{context: context, module: m}

	runtime.SetFinalizer(met, (*JITModuleMethod).finalize)

	return met, nil
}

// RunMethod executes given method with tensors or tuples as input
func (m *JITModule) RunMethod(method string, inputs ...interface{}) (interface{}, error) {
	met, err := m.GetMethod(method)
	if err != nil {
		return nil, err
	}

	return met.Run(inputs...)
}

// Forward exectures forward method of the module (forward propagation)
func (m *JITModule) Forward(inputs ...interface{}) (interface{}, error) {
	return m.RunMethod("forward", inputs...)
}

// GetMethodNames returns all method names from the module
func (m *JITModule) GetMethodNames() []string {
	var resLen C.ulong
	cnamesPtr := C.Torch_JITModuleGetMethodNames(m.context, &resLen)
	resSlice := (*[1 << 30]*C.char)(unsafe.Pointer(cnamesPtr))[:resLen:resLen]
	defer C.free(unsafe.Pointer(cnamesPtr))

	names := make([]string, len(resSlice))

	for i, name := range resSlice {
		names[i] = C.GoString(name)
		C.free(unsafe.Pointer(name))
	}

	return names
}

func (m *JITModule) finalize() {
	C.Torch_DeleteJITModule(m.context)
}

// JITModuleMethod is single method from a JITModule
type JITModuleMethod struct {
	context C.Torch_JITModuleMethodContext
	module  *JITModule
}

// Run executes given method with tensors as input
func (m *JITModuleMethod) Run(inputs ...interface{}) (interface{}, error) {
	ivalues := make([]C.Torch_IValue, len(inputs))
	for i, t := range inputs {
		var err error
		ivalues[i], err = convertGoValueToIValue(t)
		if err != nil {
			return nil, err
		}
	}

	defer freeIValues(ivalues)

	var cErr C.Torch_Error
	ival := C.Torch_JITModuleMethodRun(
		m.context,
		(*C.Torch_IValue)(&ivalues[0]),
		C.ulong(len(ivalues)),
		&cErr,
	)
	if err := checkError(cErr); err != nil {
		return nil, err
	}

	defer freeIValues([]C.Torch_IValue{ival})

	runtime.KeepAlive(inputs)

	return convertIValueToGoType(ival)
}

// Arguments returns method arguments for the method schema
func (m *JITModuleMethod) Arguments() []JITModuleMethodArgument {
	var resSize C.ulong
	resPtr := C.Torch_JITModuleMethodArguments(m.context, &resSize)
	defer C.free(unsafe.Pointer(resPtr))

	resSlice := (*[1 << 30]C.Torch_ModuleMethodArgument)(unsafe.Pointer(resPtr))[:resSize:resSize]

	args := make([]JITModuleMethodArgument, int(resSize))
	for i, arg := range resSlice {
		args[i] = JITModuleMethodArgument{
			Name: C.GoString(arg.name),
			Type: C.GoString(arg.typ),
		}
		C.free(unsafe.Pointer(arg.typ))
		C.free(unsafe.Pointer(arg.name))
	}

	return args
}

// Returns returns method return type information for the method schema
func (m *JITModuleMethod) Returns() []JITModuleMethodArgument {
	var resSize C.ulong
	resPtr := C.Torch_JITModuleMethodReturns(m.context, &resSize)
	defer C.free(unsafe.Pointer(resPtr))

	resSlice := (*[1 << 30]C.Torch_ModuleMethodArgument)(unsafe.Pointer(resPtr))[:resSize:resSize]

	args := make([]JITModuleMethodArgument, int(resSize))
	for i, arg := range resSlice {
		args[i] = JITModuleMethodArgument{
			Name: C.GoString(arg.name),
			Type: C.GoString(arg.typ),
		}
		C.free(unsafe.Pointer(arg.typ))
		C.free(unsafe.Pointer(arg.name))
	}

	return args
}

func (m *JITModuleMethod) finalize() {
	C.Torch_DeleteJITModuleMethod(m.context)
}

// JITModuleMethodArgument contains information of a single method argument
type JITModuleMethodArgument struct {
	Name string
	Type string
}

func freeTuple(tuple *C.Torch_IValueTuple) {
	valuesSlice := (*[1 << 30]C.Torch_IValue)(unsafe.Pointer(tuple.values))[:tuple.length:tuple.length]
	freeIValues(valuesSlice)
	C.free(unsafe.Pointer(tuple.values))
	C.free(unsafe.Pointer(tuple))
}

func freeIValues(values []C.Torch_IValue) {
	for _, val := range values {
		if val.itype == C.Torch_IValueTypeTuple {
			freeTuple((*C.Torch_IValueTuple)(val.data_ptr))
		}
	}
}

func convertIValueToGoType(ival C.Torch_IValue) (interface{}, error) {
	if ival.itype == C.Torch_IValueTypeTensor {
		tensorContext := (C.Torch_TensorContext)(ival.data_ptr)
		return tensorWithContext(tensorContext), nil
	} else if ival.itype == C.Torch_IValueTypeTuple {
		tuple := (*C.Torch_IValueTuple)(ival.data_ptr)
		return convertIValueTupleToTuple(tuple)
	}

	// TODO handle errors
	return nil, nil
}

func convertIValueTupleToTuple(tuple *C.Torch_IValueTuple) (Tuple, error) {
	valuesSlice := (*[1 << 30]C.Torch_IValue)(unsafe.Pointer(tuple.values))[:tuple.length:tuple.length]

	goTuple := make(Tuple, len(valuesSlice))

	for i, ival := range valuesSlice {
		var err error
		goTuple[i], err = convertIValueToGoType(ival)
		if err != nil {
			return nil, err
		}
	}

	return goTuple, nil
}

func convertGoValueToIValue(val interface{}) (C.Torch_IValue, error) {
	switch v := val.(type) {
	case *Tensor:
		return C.Torch_IValue{
			itype:    C.Torch_IValueTypeTensor,
			data_ptr: unsafe.Pointer(v.context),
		}, nil
	case Tuple:
		tuple := (*C.Torch_IValueTuple)(C.malloc(C.size_of_ivalue_tuple))
		tuple.values = (*C.Torch_IValue)(C.malloc(C.size_of_ivalue * C.ulong(len(v))))
		tuple.length = C.ulong(len(v))

		valuesSlice := (*[1 << 30]C.Torch_IValue)(unsafe.Pointer(tuple.values))[:tuple.length:tuple.length]

		for i, val := range v {
			var err error
			valuesSlice[i], err = convertGoValueToIValue(val)
			if err != nil {
				return C.Torch_IValue{}, err
			}
		}

		return C.Torch_IValue{
			itype:    C.Torch_IValueTypeTuple,
			data_ptr: unsafe.Pointer(tuple),
		}, nil
	default:
		return C.Torch_IValue{}, fmt.Errorf("invalid input type for run %T", val)
	}
}
