package torch_test

import (
	"fmt"

	"github.com/orktes/go-torch"
)

func ExampleCompileTorchScript() {
	module, _ := torch.CompileTorchScript(`
		def sum(a, b):
			return a + b
	`)

	a, _ := torch.NewTensor([]float32{1})
	b, _ := torch.NewTensor([]float32{2})

	result, _ := module.RunMethod("sum", a, b)
	fmt.Printf("[1] + [2] = %+v\n", result.(*torch.Tensor).Value())
	// output: [1] + [2] = [3]
}
