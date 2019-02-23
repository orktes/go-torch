package torch

import "testing"

const sumScript = `
def sum(a, b):
    return a + b
`

func Test_CompileTorchScript(t *testing.T) {
	module, err := CompileTorchScript(sumScript)
	if err != nil {
		t.Error(err)
	}

	method, err := module.GetMethod("sum")
	if err != nil {
		t.Error(err)
	}

	if method == nil {
		t.Error("method was nil")
	}

	a, _ := NewTensor([]float32{1, 2})
	b, _ := NewTensor([]float32{1, 2})

	res, err := method.Run(a, b)
	if err != nil {
		t.Fatal(err)
	}

	if res[0].Value().([]float32)[0] != 2 {
		t.Error("1 + 1 should equal 2 but got", res[0].Value())
	}

	if res[0].Value().([]float32)[1] != 4 {
		t.Error("2 + 2 should equal 4 but got", res[0].Value())
	}

}
