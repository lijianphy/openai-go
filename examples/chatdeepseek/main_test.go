package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/openai/openai-go/v3"
)

func TestHistoryItemJSONRoundTrip(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		item openai.ChatCompletionMessageParamUnion
	}{
		{
			name: "system_message",
			item: openai.SystemMessage(systemPrompt),
		},
		{
			name: "user_message",
			item: openai.UserMessage("hello"),
		},
		{
			name: "assistant_tool_call",
			item: mustHistoryItem(t, `{
				"role":"assistant",
				"content":"Checking",
				"tool_calls":[
					{
						"id":"call_123",
						"type":"function",
						"function":{
							"name":"bash",
							"arguments":"{\"command\":\"pwd\"}"
						}
					}
				]
			}`),
		},
		{
			name: "tool_message",
			item: openai.ToolMessage(`{"stdout":"ok"}`, "call_123"),
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			data, err := marshalHistoryItemJSON(tt.item)
			if err != nil {
				t.Fatalf("marshalHistoryItemJSON() error = %v", err)
			}

			roundTripped, err := unmarshalHistoryItemJSON(data)
			if err != nil {
				t.Fatalf("unmarshalHistoryItemJSON() error = %v", err)
			}

			roundTripData, err := json.Marshal(roundTripped)
			if err != nil {
				t.Fatalf("json.Marshal(roundTripped) error = %v", err)
			}

			if err := ensureJSONRoundTrip(data, roundTripData); err != nil {
				t.Fatalf("ensureJSONRoundTrip() error = %v", err)
			}
		})
	}
}

func TestChatHistoryJSONLLoadAndAppend(t *testing.T) {
	t.Parallel()

	historyPath := filepath.Join(t.TempDir(), "history", "session.jsonl")
	history, err := newChatHistory(historyPath)
	if err != nil {
		t.Fatalf("newChatHistory() error = %v", err)
	}

	items := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(systemPrompt),
		openai.UserMessage("Run pwd"),
		mustHistoryItem(t, `{
			"role":"assistant",
			"content":"Checking",
			"tool_calls":[
				{
					"id":"call_456",
					"type":"function",
					"function":{
						"name":"bash",
						"arguments":"{\"command\":\"pwd\"}"
					}
				}
			]
		}`),
		openai.ToolMessage(`{"stdout":"/tmp\n"}`, "call_456"),
		openai.AssistantMessage("done"),
	}

	for _, item := range items {
		if err := history.Append(item); err != nil {
			t.Fatalf("Append() error = %v", err)
		}
	}

	if err := history.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	rawFile, err := os.ReadFile(historyPath)
	if err != nil {
		t.Fatalf("os.ReadFile() error = %v", err)
	}
	if got := bytes.Count(rawFile, []byte("\n")); got != len(items) {
		t.Fatalf("newline count = %d, want %d", got, len(items))
	}

	loaded, err := loadHistoryFile(historyPath)
	if err != nil {
		t.Fatalf("loadHistoryFile() error = %v", err)
	}
	if len(loaded) != len(items) {
		t.Fatalf("len(loaded) = %d, want %d", len(loaded), len(items))
	}

	for i := range items {
		want, err := json.Marshal(items[i])
		if err != nil {
			t.Fatalf("json.Marshal(items[%d]) error = %v", i, err)
		}
		got, err := json.Marshal(loaded[i])
		if err != nil {
			t.Fatalf("json.Marshal(loaded[%d]) error = %v", i, err)
		}
		if err := ensureJSONRoundTrip(want, got); err != nil {
			t.Fatalf("history item %d mismatch: %v", i, err)
		}
	}
}

func mustHistoryItem(t *testing.T, raw string) openai.ChatCompletionMessageParamUnion {
	t.Helper()

	var item openai.ChatCompletionMessageParamUnion
	if err := json.Unmarshal([]byte(raw), &item); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	return item
}

func ensureJSONRoundTrip(original []byte, roundTrip []byte) error {
	canonicalOriginal, err := canonicalJSON(original)
	if err != nil {
		return err
	}

	canonicalRoundTrip, err := canonicalJSON(roundTrip)
	if err != nil {
		return err
	}

	if !bytes.Equal(canonicalOriginal, canonicalRoundTrip) {
		return fmt.Errorf("json mismatch: original=%s round_trip=%s", canonicalOriginal, canonicalRoundTrip)
	}

	return nil
}

func canonicalJSON(data []byte) ([]byte, error) {
	var value any
	if err := json.Unmarshal(data, &value); err != nil {
		return nil, err
	}

	return json.Marshal(value)
}
