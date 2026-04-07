package ssestream

import (
	"io"
	"net/http"
	"strings"
	"testing"
)

type testEvent struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

func TestStreamIgnoresLeadingBlankLines(t *testing.T) {
	stream := NewStream[testEvent](NewDecoder(newTestEventStreamResponse("\n\nevent: message\ndata: {\"type\":\"message\",\"message\":\"hello\"}\n\n")), nil)
	defer func() { _ = stream.Close() }()

	if !stream.Next() {
		t.Fatalf("expected stream event, got err: %v", stream.Err())
	}

	if got := stream.Current(); got.Type != "message" || got.Message != "hello" {
		t.Fatalf("unexpected event: %#v", got)
	}

	if stream.Next() {
		t.Fatal("expected stream to be exhausted")
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}
}

func TestStreamFlushesFinalEventAtEOF(t *testing.T) {
	stream := NewStream[testEvent](NewDecoder(newTestEventStreamResponse("event: message\ndata: {\"type\":\"message\",\"message\":\"hello\"}\n")), nil)
	defer func() { _ = stream.Close() }()

	if !stream.Next() {
		t.Fatalf("expected stream event, got err: %v", stream.Err())
	}

	if got := stream.Current(); got.Type != "message" || got.Message != "hello" {
		t.Fatalf("unexpected event: %#v", got)
	}

	if stream.Next() {
		t.Fatal("expected stream to be exhausted")
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}
}

func newTestEventStreamResponse(body string) *http.Response {
	return &http.Response{
		Header: http.Header{
			"Content-Type": []string{"text/event-stream"},
		},
		Body: io.NopCloser(strings.NewReader(body)),
	}
}
