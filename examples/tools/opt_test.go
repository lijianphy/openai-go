package tools

import (
	"encoding/json"
	"testing"
)

type optJSONFixture struct {
	Name   Opt[string] `json:"name,omitzero"`
	Age    Opt[int]    `json:"age,omitzero"`
	Active Opt[bool]   `json:"active,omitzero"`
}

func TestOptStateHelpers(t *testing.T) {
	opt := None[int]()
	if opt.IsSet() {
		t.Fatal("None().IsSet() = true, want false")
	}
	if !opt.IsZero() {
		t.Fatal("None().IsZero() = false, want true")
	}

	value, ok := opt.Get()
	if ok {
		t.Fatal("None().Get() ok = true, want false")
	}
	if value != 0 {
		t.Fatalf("None().Get() value = %d, want 0", value)
	}

	opt = Some(7)
	if !opt.IsSet() {
		t.Fatal("Some(7).IsSet() = false, want true")
	}
	if opt.IsZero() {
		t.Fatal("Some(7).IsZero() = true, want false")
	}

	value, ok = opt.Get()
	if !ok {
		t.Fatal("Some(7).Get() ok = false, want true")
	}
	if value != 7 {
		t.Fatalf("Some(7).Get() value = %d, want 7", value)
	}

	opt.Set(9)
	value, ok = opt.Get()
	if !ok {
		t.Fatal("Set(9) left option unset")
	}
	if value != 9 {
		t.Fatalf("after Set(9), Get() value = %d, want 9", value)
	}

	opt.Clear()
	if opt.IsSet() {
		t.Fatal("after Clear(), IsSet() = true, want false")
	}
	if !opt.IsZero() {
		t.Fatal("after Clear(), IsZero() = false, want true")
	}
}

func TestOptMarshalJSON(t *testing.T) {
	tests := []struct {
		name string
		opt  any
		want string
	}{
		{
			name: "unset encodes as null",
			opt:  None[string](),
			want: "null",
		},
		{
			name: "set string encodes as string",
			opt:  Some("alice"),
			want: `"alice"`,
		},
		{
			name: "set zero value encodes as value",
			opt:  Some(0),
			want: "0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.opt)
			if err != nil {
				t.Fatalf("json.Marshal() error = %v", err)
			}
			if string(data) != tt.want {
				t.Fatalf("json.Marshal() = %s, want %s", data, tt.want)
			}
		})
	}
}

func TestOptMarshalWithOmitZero(t *testing.T) {
	fixture := optJSONFixture{
		Age:    Some(0),
		Active: Some(false),
	}

	data, err := json.Marshal(fixture)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	const want = `{"age":0,"active":false}`
	if string(data) != want {
		t.Fatalf("json.Marshal() = %s, want %s", data, want)
	}
}

func TestOptUnmarshalJSON(t *testing.T) {
	t.Run("present value sets option", func(t *testing.T) {
		var opt Opt[int]
		if err := json.Unmarshal([]byte("0"), &opt); err != nil {
			t.Fatalf("json.Unmarshal() error = %v", err)
		}

		value, ok := opt.Get()
		if !ok {
			t.Fatal("Get() ok = false, want true")
		}
		if value != 0 {
			t.Fatalf("Get() value = %d, want 0", value)
		}
	})

	t.Run("null clears option", func(t *testing.T) {
		opt := Some("alice")
		if err := json.Unmarshal([]byte("null"), &opt); err != nil {
			t.Fatalf("json.Unmarshal() error = %v", err)
		}
		if opt.IsSet() {
			t.Fatal("IsSet() = true after null, want false")
		}
	})

	t.Run("invalid value returns error", func(t *testing.T) {
		var opt Opt[int]
		if err := json.Unmarshal([]byte(`"bad"`), &opt); err == nil {
			t.Fatal("json.Unmarshal() error = nil, want non-nil")
		}
	})
}

func TestOptUnmarshalStructFieldBehavior(t *testing.T) {
	t.Run("missing field remains unset", func(t *testing.T) {
		var fixture optJSONFixture
		if err := json.Unmarshal([]byte(`{"name":"alice"}`), &fixture); err != nil {
			t.Fatalf("json.Unmarshal() error = %v", err)
		}

		name, ok := fixture.Name.Get()
		if !ok || name != "alice" {
			t.Fatalf("Name.Get() = (%q, %v), want (%q, true)", name, ok, "alice")
		}
		if fixture.Age.IsSet() {
			t.Fatal("Age.IsSet() = true, want false")
		}
		if fixture.Active.IsSet() {
			t.Fatal("Active.IsSet() = true, want false")
		}
	})

	t.Run("null clears previously set field", func(t *testing.T) {
		fixture := optJSONFixture{
			Name: Some("alice"),
			Age:  Some(9),
		}

		if err := json.Unmarshal([]byte(`{"name":null,"age":0}`), &fixture); err != nil {
			t.Fatalf("json.Unmarshal() error = %v", err)
		}

		if fixture.Name.IsSet() {
			t.Fatal("Name.IsSet() = true, want false")
		}

		age, ok := fixture.Age.Get()
		if !ok || age != 0 {
			t.Fatalf("Age.Get() = (%d, %v), want (0, true)", age, ok)
		}
	})
}
