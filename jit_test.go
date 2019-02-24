package torch

import (
	"io/ioutil"
	"os"
	"path"
	"testing"
)

const sumScript = `
def sum(a, b):
    return a + b
`

const sumAndSubs = `
def sum_sub(a, b):
	return (a + b, a - b)
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

	args := method.Arguments()
	if args[0].Name != "a" {
		t.Error("wrong arg name returned", args[0].Name)
	}
	if args[1].Name != "b" {
		t.Error("wrong arg name returned", args[1].Name)
	}

	returns := method.Returns()

	if len(returns) != 1 {
		t.Error("sum should only return one tensor")
	}

	a, _ := NewTensor([]float32{1, 2})
	b, _ := NewTensor([]float32{1, 2})

	res, err := method.Run(a, b)
	if err != nil {
		t.Fatal(err)
	}

	if res.(*Tensor).Value().([]float32)[0] != 2 {
		t.Error("1 + 1 should equal 2 but got", res.(*Tensor).Value())
	}

	if res.(*Tensor).Value().([]float32)[1] != 4 {
		t.Error("2 + 2 should equal 4 but got", res.(*Tensor).Value())
	}

}

func Test_TupleReturn(t *testing.T) {
	module, err := CompileTorchScript(sumAndSubs)
	if err != nil {
		t.Error(err)
	}

	a, _ := NewTensor([]float32{1})
	b, _ := NewTensor([]float32{1})

	res, err := module.RunMethod("sum_sub", a, b)
	if err != nil {
		t.Fatal(err)
	}

	sum := res.(Tuple).Get(0).(*Tensor)
	if sum.Value().([]float32)[0] != 2 {
		t.Error("1 + 1 should equal 2 but got", res.(*Tensor).Value())
	}

	sub := res.(Tuple).Get(1).(*Tensor)
	if sub.Value().([]float32)[0] != 0 {
		t.Error("1 - 1 should equal 0 but got", res.(*Tensor).Value())
	}

}
func Test_SaveAndLoadJITModule(t *testing.T) {
	dir, err := ioutil.TempDir("", "modules")
	if err != nil {
		t.Fatal(err)
	}

	defer os.RemoveAll(dir) // clean up

	module, err := CompileTorchScript(sumScript)
	if err != nil {
		t.Error(err)
	}

	err = module.Save(path.Join(dir, "sum.pt"))
	if err != nil {
		t.Error(err)
	}

	loaded, err := LoadJITModule(path.Join(dir, "sum.pt"))
	if err != nil {
		t.Error(err)
	}

	a, _ := NewTensor([]float32{1, 2})
	b, _ := NewTensor([]float32{1, 2})

	res, err := loaded.RunMethod("sum", a, b)
	if err != nil {
		t.Fatal(err)
	}

	if res.(*Tensor).Value().([]float32)[0] != 2 {
		t.Error("1 + 1 should equal 2 but got", res.(*Tensor).Value())
	}

	if res.(*Tensor).Value().([]float32)[1] != 4 {
		t.Error("2 + 2 should equal 4 but got", res.(*Tensor).Value())
	}
}
