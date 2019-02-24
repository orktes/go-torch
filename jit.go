package torch

// #include "torch.hpp"
// #include <stdlib.h>
import "C"
import (
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

	ctx := C.Torch_CompileTorchScript(cstr)
	mod := &JITModule{context: ctx}
	runtime.SetFinalizer(mod, (*JITModule).finalize)

	return mod, nil
}

// LoadJITModule loads module from file
func LoadJITModule(path string) (*JITModule, error) {
	cstr := C.CString(path)
	defer C.free(unsafe.Pointer(cstr))

	ctx := C.Torch_LoadJITModule(cstr)
	mod := &JITModule{context: ctx}
	runtime.SetFinalizer(mod, (*JITModule).finalize)
	// TODO handle errors
	return mod, nil
}

// Save saves Module to given path
func (m *JITModule) Save(path string) error {
	cstr := C.CString(path)
	defer C.free(unsafe.Pointer(cstr))

	C.Torch_ExportJITModule(m.context, cstr)

	// TODO handle errors
	return nil
}

// GetMethod returns a method from a JITModule
func (m *JITModule) GetMethod(method string) (*JITModuleMethod, error) {
	cstr := C.CString(method)
	defer C.free(unsafe.Pointer(cstr))

	context := C.Torch_JITModuleGetMethod(m.context, cstr)
	met := &JITModuleMethod{context: context, module: m}

	runtime.SetFinalizer(met, (*JITModuleMethod).finalize)

	return met, nil
}

// RunMethod executes given method with tensors as input
func (m *JITModule) RunMethod(method string, inputs ...*Tensor) ([]*Tensor, error) {
	met, err := m.GetMethod(method)
	if err != nil {
		return nil, err
	}

	return met.Run(inputs...)
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
func (m *JITModuleMethod) Run(inputs ...*Tensor) ([]*Tensor, error) {
	contexts := make([]C.Torch_TensorContext, len(inputs))
	for i, t := range inputs {
		contexts[i] = t.context
	}

	var resSize C.ulong
	resPtr := C.Torch_JITModuleMethodRun(
		m.context,
		(*C.Torch_TensorContext)(&contexts[0]),
		C.ulong(len(contexts)),
		&resSize)

	if resSize == 0 || resPtr == nil {
		return nil, nil
	}

	defer C.free(unsafe.Pointer(resPtr))

	ctxSlice := (*[1 << 30]C.Torch_TensorContext)(unsafe.Pointer(resPtr))[:resSize:resSize]
	outputs := make([]*Tensor, len(ctxSlice))
	for i, ctx := range ctxSlice {
		outputs[i] = tensorWithContext(ctx)
	}

	runtime.KeepAlive(inputs)

	return outputs, nil
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
		}
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
		}
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
}
