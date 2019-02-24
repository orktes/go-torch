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

// GetMethod returns a method from a JITModule
func (m *JITModule) GetMethod(method string) (*JITModuleMethod, error) {
	cstr := C.CString(method)
	defer C.free(unsafe.Pointer(cstr))

	context := C.Torch_JITModuleGetMethod(m.context, cstr)
	met := &JITModuleMethod{context: context, module: m}

	runtime.SetFinalizer(met, (*JITModuleMethod).finalize)

	return met, nil
}

func (m *JITModule) finalize() {
	C.Torch_DeleteJITModule(m.context)
}

// JITModuleMethod is single method from a JITModule
type JITModuleMethod struct {
	context C.Torch_JITModuleMethodContext
	module  *JITModule
}

// Run executes given method given tensors as input
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

	ctxSlice := (*[1 << 30]C.Torch_TensorContext)(unsafe.Pointer(resPtr))[:resSize:resSize]
	outputs := make([]*Tensor, len(ctxSlice))
	for i, ctx := range ctxSlice {
		outputs[i] = tensorWithContext(ctx)
	}

	runtime.KeepAlive(inputs)

	return outputs, nil
}

func (m *JITModuleMethod) finalize() {
	C.Torch_DeleteJITModuleMethod(m.context)
}
