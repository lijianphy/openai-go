package tools

import (
	"bytes"
	"encoding/json"
)

// Opt stores an optional value and works with json:",omitzero" to omit unset
// fields without using pointers. Missing fields and explicit JSON null both
// decode to the unset state.
type Opt[T any] struct {
	value T
	set   bool
}

// Some returns an Opt with a value present.
func Some[T any](value T) Opt[T] {
	return Opt[T]{
		value: value,
		set:   true,
	}
}

// None returns an Opt with no value present.
func None[T any]() Opt[T] {
	return Opt[T]{}
}

// IsZero reports whether the option is unset so json:",omitzero" can omit it.
func (o Opt[T]) IsZero() bool {
	return !o.set
}

// IsSet reports whether the option contains a value.
func (o Opt[T]) IsSet() bool {
	return o.set
}

// Get returns the value and whether it is present.
func (o Opt[T]) Get() (T, bool) {
	return o.value, o.set
}

// Set stores a value and marks the option as present.
func (o *Opt[T]) Set(value T) {
	o.value = value
	o.set = true
}

// Clear removes the value and marks the option as unset.
func (o *Opt[T]) Clear() {
	var zero T
	o.value = zero
	o.set = false
}

// MarshalJSON encodes the contained value. Unset values encode as null when
// marshaled directly; struct fields tagged with json:",omitzero" are omitted.
func (o Opt[T]) MarshalJSON() ([]byte, error) {
	if !o.set {
		return []byte("null"), nil
	}

	return json.Marshal(o.value)
}

// UnmarshalJSON decodes a present JSON value. Missing fields leave the option
// untouched because UnmarshalJSON is not called. JSON null is treated as unset.
func (o *Opt[T]) UnmarshalJSON(data []byte) error {
	data = bytes.TrimSpace(data)
	if bytes.Equal(data, []byte("null")) {
		o.Clear()
		return nil
	}

	if err := json.Unmarshal(data, &o.value); err != nil {
		return err
	}

	o.set = true
	return nil
}
