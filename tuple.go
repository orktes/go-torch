package torch

// Tuple a tuple type
type Tuple []interface{}

// NewTuple returns a new tuple for given values (go types, torch.Tensor, torch.Tuple)
func NewTuple(vals ...interface{}) (Tuple, error) {
	var err error
	tuple := make(Tuple, len(vals))
	for i, val := range vals {
		switch val.(type) {
		case *Tensor, Tuple:
			tuple[i] = val
		default:
			tuple[i], err = NewTensor(val)
			if err != nil {
				return nil, err
			}
		}
	}
	return tuple, nil
}

// Get returns a type in specific tuple index (otherwise returns nil)
func (t Tuple) Get(index int) interface{} {
	if len(t)-1 >= index {
		return t[index]
	}

	return nil
}
