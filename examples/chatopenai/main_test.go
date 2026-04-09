package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/openai/openai-go/v3/responses"
)

func TestHistoryItemJSONRoundTrip(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		item responses.ResponseInputItemUnionParam
	}{
		{
			name: "system_message",
			item: responses.ResponseInputItemParamOfMessage(systemPrompt, responses.EasyInputMessageRoleSystem),
		},
		{
			name: "function_call_output",
			item: responses.ResponseInputItemParamOfFunctionCallOutput("call_123", `{"stdout":"ok"}`),
		},
		{
			name: "output_message_replay",
			item: mustReplayOutputMessageItem(t, `{
				"id":"msg_123",
				"type":"message",
				"role":"assistant",
				"status":"completed",
				"phase":"final_answer",
				"content":[
					{
						"type":"output_text",
						"text":"hello",
						"annotations":[]
					}
				]
			}`),
		},
		{
			name: "reasoning_replay",
			item: mustReplayReasoningItem(t, `{
				"id":"rs_123",
				"type":"reasoning",
				"status":"completed",
				"summary":[
					{
						"type":"summary_text",
						"text":"Checked the tool output."
					}
				],
				"encrypted_content":"ciphertext"
			}`),
		},
		{
			name: "function_call_replay",
			item: mustReplayFunctionCallItem(t, `{
				"id":"fc_123",
				"type":"function_call",
				"status":"completed",
				"call_id":"call_123",
				"name":"bash",
				"arguments":"{\"command\":\"pwd\"}"
			}`),
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

func TestHistoryInstructionJSONRoundTrip(t *testing.T) {
	t.Parallel()

	data, err := marshalHistoryInstructionJSON(systemPrompt)
	if err != nil {
		t.Fatalf("marshalHistoryInstructionJSON() error = %v", err)
	}

	instruction, ok, err := unmarshalHistoryInstructionJSON(data)
	if err != nil {
		t.Fatalf("unmarshalHistoryInstructionJSON() error = %v", err)
	}
	if !ok {
		t.Fatalf("unmarshalHistoryInstructionJSON() ok = false, want true")
	}
	if instruction != systemPrompt {
		t.Fatalf("instruction = %q, want %q", instruction, systemPrompt)
	}
}

func TestChatHistoryJSONLLoadAndAppend(t *testing.T) {
	t.Parallel()

	historyPath := filepath.Join(t.TempDir(), "history", "session.jsonl")
	history, err := newChatHistory(historyPath)
	if err != nil {
		t.Fatalf("newChatHistory() error = %v", err)
	}

	items := []responses.ResponseInputItemUnionParam{
		mustReplayFunctionCallItem(t, `{
			"id":"fc_456",
			"type":"function_call",
			"status":"completed",
			"call_id":"call_456",
			"name":"bash",
			"arguments":"{\"command\":\"echo hi\"}"
		}`),
		responses.ResponseInputItemParamOfFunctionCallOutput("call_456", `{"stdout":"hi\n"}`),
		mustReplayOutputMessageItem(t, `{
			"id":"msg_456",
			"type":"message",
			"role":"assistant",
			"status":"completed",
			"phase":"commentary",
			"content":[
				{
					"type":"output_text",
					"text":"done",
					"annotations":[]
				}
			]
		}`),
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
	if got := bytes.Count(rawFile, []byte("\n")); got != len(items)+1 {
		t.Fatalf("newline count = %d, want %d", got, len(items)+1)
	}

	lines := bytes.Split(bytes.TrimSpace(rawFile), []byte("\n"))
	if len(lines) != len(items)+1 {
		t.Fatalf("line count = %d, want %d", len(lines), len(items)+1)
	}

	instruction, ok, err := unmarshalHistoryInstructionJSON(lines[0])
	if err != nil {
		t.Fatalf("unmarshalHistoryInstructionJSON(first line) error = %v", err)
	}
	if !ok {
		t.Fatalf("first line is not an instruction record: %s", lines[0])
	}
	if instruction != systemPrompt {
		t.Fatalf("instruction = %q, want %q", instruction, systemPrompt)
	}

	instruction, loaded, err := loadHistoryFile(historyPath)
	if err != nil {
		t.Fatalf("loadHistoryFile() error = %v", err)
	}
	if instruction != systemPrompt {
		t.Fatalf("instruction = %q, want %q", instruction, systemPrompt)
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

func TestLoadHistoryFileUsesDefaultInstructionWhenFirstLineIsEmpty(t *testing.T) {
	t.Parallel()

	item := responses.ResponseInputItemParamOfMessage("hello", responses.EasyInputMessageRoleUser)
	itemJSON, err := marshalHistoryItemJSON(item)
	if err != nil {
		t.Fatalf("marshalHistoryItemJSON() error = %v", err)
	}

	historyPath := filepath.Join(t.TempDir(), "history.jsonl")
	if err := os.WriteFile(historyPath, append([]byte("\n"), append(itemJSON, '\n')...), 0o600); err != nil {
		t.Fatalf("os.WriteFile() error = %v", err)
	}

	instruction, items, err := loadHistoryFile(historyPath)
	if err != nil {
		t.Fatalf("loadHistoryFile() error = %v", err)
	}
	if instruction != systemPrompt {
		t.Fatalf("instruction = %q, want %q", instruction, systemPrompt)
	}
	if len(items) != 1 {
		t.Fatalf("len(items) = %d, want 1", len(items))
	}
}

func TestChatHistoryAppendDoesNotRewriteExistingFirstLine(t *testing.T) {
	t.Parallel()

	firstLine := []byte(` { "type" : "instruction" , "content" : "custom instruction" } ` + "\n")
	existingItem := responses.ResponseInputItemParamOfMessage("hello", responses.EasyInputMessageRoleUser)
	existingItemJSON, err := marshalHistoryItemJSON(existingItem)
	if err != nil {
		t.Fatalf("marshalHistoryItemJSON(existing) error = %v", err)
	}

	historyPath := filepath.Join(t.TempDir(), "history.jsonl")
	initialFile := append(append([]byte(nil), firstLine...), append(existingItemJSON, '\n')...)
	if err := os.WriteFile(historyPath, initialFile, 0o600); err != nil {
		t.Fatalf("os.WriteFile() error = %v", err)
	}

	history, err := newChatHistory(historyPath)
	if err != nil {
		t.Fatalf("newChatHistory() error = %v", err)
	}
	defer history.Close()

	if history.instruction != "custom instruction" {
		t.Fatalf("history.instruction = %q, want %q", history.instruction, "custom instruction")
	}

	newItem := responses.ResponseInputItemParamOfMessage("world", responses.EasyInputMessageRoleUser)
	if err := history.Append(newItem); err != nil {
		t.Fatalf("Append() error = %v", err)
	}
	if err := history.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	rawFile, err := os.ReadFile(historyPath)
	if err != nil {
		t.Fatalf("os.ReadFile() error = %v", err)
	}
	if !bytes.HasPrefix(rawFile, firstLine) {
		t.Fatalf("first line was rewritten: got prefix %q, want %q", bytes.SplitN(rawFile, []byte("\n"), 2)[0], bytes.TrimSuffix(firstLine, []byte("\n")))
	}

	lines := bytes.Split(bytes.TrimRight(rawFile, "\n"), []byte("\n"))
	if len(lines) != 3 {
		t.Fatalf("line count = %d, want 3", len(lines))
	}
	if !bytes.Equal(lines[1], existingItemJSON) {
		t.Fatalf("line 2 = %s, want %s", lines[1], existingItemJSON)
	}

	newItemJSON, err := marshalHistoryItemJSON(newItem)
	if err != nil {
		t.Fatalf("marshalHistoryItemJSON(new) error = %v", err)
	}
	if !bytes.Equal(lines[2], newItemJSON) {
		t.Fatalf("line 3 = %s, want %s", lines[2], newItemJSON)
	}
}

func mustReplayOutputMessageItem(t *testing.T, raw string) responses.ResponseInputItemUnionParam {
	t.Helper()

	item := mustResponseOutputItemUnion(t, raw)
	messageParam := item.AsMessage().ToParam()
	return responses.ResponseInputItemUnionParam{
		OfOutputMessage: &messageParam,
	}
}

func mustReplayReasoningItem(t *testing.T, raw string) responses.ResponseInputItemUnionParam {
	t.Helper()

	item := mustResponseOutputItemUnion(t, raw)
	reasoningParam := item.AsReasoning().ToParam()
	return responses.ResponseInputItemUnionParam{
		OfReasoning: &reasoningParam,
	}
}

func mustReplayFunctionCallItem(t *testing.T, raw string) responses.ResponseInputItemUnionParam {
	t.Helper()

	item := mustResponseOutputItemUnion(t, raw)
	functionCallParam := item.AsFunctionCall().ToParam()
	return responses.ResponseInputItemUnionParam{
		OfFunctionCall: &functionCallParam,
	}
}

func mustResponseOutputItemUnion(t *testing.T, raw string) responses.ResponseOutputItemUnion {
	t.Helper()

	var item responses.ResponseOutputItemUnion
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
