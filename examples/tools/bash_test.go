package tools

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/openai/openai-go/v3/responses"
)

func TestNewRegistryIncludesBashTool(t *testing.T) {
	registry := NewRegistry()
	definitions := registry.Definitions()
	if len(definitions) != 1 {
		t.Fatalf("Definitions() length = %d, want 1", len(definitions))
	}

	fn := definitions[0].OfFunction
	if fn == nil {
		t.Fatal("Definitions()[0] is not a function tool")
	}
	if fn.Name != BashToolName {
		t.Fatalf("tool name = %q, want %q", fn.Name, BashToolName)
	}
}

func TestRegistryExecuteBashTool(t *testing.T) {
	registry := NewRegistry()

	item := registry.Execute(context.Background(), responses.ResponseFunctionToolCall{
		Name:      BashToolName,
		CallID:    "call_123",
		Arguments: `{"script":"printf hello"}`,
	})

	if item.OfFunctionCallOutput == nil {
		t.Fatal("Execute() did not return a function_call_output item")
	}
	if item.OfFunctionCallOutput.CallID != "call_123" {
		t.Fatalf("CallID = %q, want %q", item.OfFunctionCallOutput.CallID, "call_123")
	}

	result := decodeFunctionCallOutput(t, item.OfFunctionCallOutput.Output)
	if result.ExitCode != 0 {
		t.Fatalf("exit_code = %d, want 0", result.ExitCode)
	}
	if result.Stdout != "hello" {
		t.Fatalf("stdout = %q, want %q", result.Stdout, "hello")
	}
	if result.Stderr != "" {
		t.Fatalf("stderr = %q, want empty", result.Stderr)
	}
}

func TestRegistryExecuteBashToolWithInvalidArguments(t *testing.T) {
	registry := NewRegistry()

	item := registry.Execute(context.Background(), responses.ResponseFunctionToolCall{
		Name:      BashToolName,
		CallID:    "call_456",
		Arguments: `{"script":123}`,
	})

	if item.OfFunctionCallOutput == nil {
		t.Fatal("Execute() did not return a function_call_output item")
	}

	result := decodeFunctionCallOutput(t, item.OfFunctionCallOutput.Output)
	if result.ExitCode != -1 {
		t.Fatalf("exit_code = %d, want -1", result.ExitCode)
	}
	if result.Error == "" {
		t.Fatal("expected an error message for invalid arguments")
	}
}

func decodeFunctionCallOutput(t *testing.T, output responses.ResponseInputItemFunctionCallOutputOutputUnionParam) bashToolResult {
	t.Helper()

	data, err := json.Marshal(output)
	if err != nil {
		t.Fatalf("json.Marshal(output) error = %v", err)
	}

	var raw string
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("json.Unmarshal(raw output) error = %v", err)
	}

	var result bashToolResult
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		t.Fatalf("json.Unmarshal(tool result) error = %v", err)
	}

	return result
}
