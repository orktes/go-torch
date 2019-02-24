WORK IN PROGRESS... USE AT OWN RISK :-)

[![Build Status](https://travis-ci.org/orktes/go-torch.svg?branch=master)](https://travis-ci.org/orktes/go-torch)
[![GoDoc](https://godoc.org/github.com/orktes/go-torch?status.svg)](http://godoc.org/github.com/orktes/go-torch)

# go-torch

LibTorch (PyTorch) bindings for Golang. Library is first and foremost designed for running inference against serialized models exported from Python version of PyTorch but can also be used to compile TorchScript applications directly for Golang.

## Installing
```sh
$ go get github.com/orktes/go-torch
```

## Usage

go-torch depends on the LibTorch shared library to be available. For more information refer to https://pytorch.org/cppdocs/ and an example [Dockerfile](https://github.com/orktes/go-torch/blob/master/scripts/Dockerfile) which is used for executing tests for the library.

```go
import (
    "github.com/orktes/go-torch"
)
```

### Using serialized PyTorch models

For instructions on how to export models for PyTorch refer to the [PyTorch documentation](https://pytorch.org/tutorials/advanced/cpp_export.html)


```go
// Load model
module, _ := torch.LoadJITModule("model.pt")

// Create an input tensor
inputTensor, _ := torch.NewTensor([][]float32{
    []float32{1, 2, 3},
})

// Forward propagation
res, _ := module.Forward(inputTensor)

```

### Using TorchScript

[TorchScript documentation](https://pytorch.org/docs/stable/jit.html)

Currently supported input and output types
- Tensor
- Tuple (of Tensor and/or nested Tuples)


```go
sumScript = `
def sum(a, b):
    return a + b
`

// Compile TorchScript
module, _ := torch.CompileTorchScript(sumScript)

// Create inputs
a, _ := torch.NewTensor([]float32{1})
b, _ := torch.NewTensor([]float32{2})

res, _ := module.RunMethod("sum", a, b)
fmt.Printf("[1] + [2] = %+v\n", res.(*torch.Tensor).Value())
// output: [1] + [2] = [3]

```

## Acknowledgements

Lots of the functionality related to converting Golang types on PyTorch Tensors are a shameless copy on what Google is doing with their Go Tensorflow bindings so part of the credit definetely goes The TensorFlow Authors.

## LICENSE

[See here](https://github.com/orktes/go-torch/blob/master/LICENSE)