package torch

// Tuple a tuple type
type Tuple []interface{}

// Get returns a type in specific tuple index (otherwise returns nil)
func (t Tuple) Get(index int) interface{} {
	if len(t)-1 >= index {
		return t[index]
	}

	return nil
}
