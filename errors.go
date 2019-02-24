package torch

// #include "torch.hpp"
// #include <stdlib.h>
import "C"
import (
	"unsafe"
)

// Error errors returned by torch functions
type Error struct {
	message string
}

func (te *Error) Error() string {
	return te.message
}

func checkError(err C.Torch_Error) *Error {
	if err.message != nil {
		defer C.free(unsafe.Pointer(err.message))
		return &Error{
			message: C.GoString(err.message),
		}
	}

	return nil
}
