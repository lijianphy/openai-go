package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/openai/openai-go/examples/tools"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

const (
	defaultBaseURL      = "https://api.deepseek.com/v1"
	defaultModel        = "deepseek-reasoner"
	systemPrompt        = "You are a helpful chatbot. Keep replies concise and clear. Use available tools when they help answer the user."
	maxToolRounds       = 8
	scannerInitialBytes = 64 * 1024
	scannerMaxBytes     = 1024 * 1024
	historyLineMaxBytes = 8 * 1024 * 1024
)

type streamedCompletion struct {
	completion openai.ChatCompletion
}

var authorizationHeaderPattern = regexp.MustCompile(`(?im)^(Authorization:\s*Bearer\s+)[^\r\n]+`)

type redactingLogWriter struct {
	file *os.File
}

type chatHistory struct {
	file       *os.File
	path       string
	items      []openai.ChatCompletionMessageParamUnion
	loadedItem int
}

func (w *redactingLogWriter) Write(p []byte) (int, error) {
	redacted := authorizationHeaderPattern.ReplaceAll(p, []byte("${1}[REDACTED]"))
	if _, err := w.file.Write(redacted); err != nil {
		return 0, err
	}

	return len(p), nil
}

func newChatHistory(path string) (*chatHistory, error) {
	items, err := loadHistoryFile(path)
	if err != nil {
		return nil, err
	}

	history := &chatHistory{
		path:       path,
		items:      items,
		loadedItem: len(items),
	}
	if path == "" {
		return history, nil
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("create history directory: %w", err)
	}

	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, fmt.Errorf("open history file %q: %w", path, err)
	}
	history.file = file

	return history, nil
}

func (h *chatHistory) Close() error {
	if h == nil || h.file == nil {
		return nil
	}

	return h.file.Close()
}

func (h *chatHistory) Len() int {
	return len(h.items)
}

func (h *chatHistory) LoadedLen() int {
	return h.loadedItem
}

func (h *chatHistory) Items() []openai.ChatCompletionMessageParamUnion {
	return append([]openai.ChatCompletionMessageParamUnion(nil), h.items...)
}

func (h *chatHistory) Append(item openai.ChatCompletionMessageParamUnion) error {
	if h.file != nil {
		if err := appendHistoryItemJSONL(h.file, item); err != nil {
			return err
		}
		if err := h.file.Sync(); err != nil {
			return fmt.Errorf("sync history file %q: %w", h.path, err)
		}
	}

	h.items = append(h.items, item)
	return nil
}

func loadHistoryFile(path string) ([]openai.ChatCompletionMessageParamUnion, error) {
	if path == "" {
		return nil, nil
	}

	file, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("open history file %q: %w", path, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, scannerInitialBytes), historyLineMaxBytes)

	var items []openai.ChatCompletionMessageParamUnion
	for lineNumber := 1; scanner.Scan(); lineNumber++ {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}

		item, err := unmarshalHistoryItemJSON(append([]byte(nil), line...))
		if err != nil {
			return nil, fmt.Errorf("parse history file %q line %d: %w", path, lineNumber, err)
		}
		items = append(items, item)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read history file %q: %w", path, err)
	}

	return items, nil
}

func appendHistoryItemJSONL(file *os.File, item openai.ChatCompletionMessageParamUnion) error {
	line, err := marshalHistoryItemJSON(item)
	if err != nil {
		return err
	}

	if _, err := file.Write(append(line, '\n')); err != nil {
		return fmt.Errorf("write history file %q: %w", file.Name(), err)
	}

	return nil
}

func marshalHistoryItemJSON(item openai.ChatCompletionMessageParamUnion) ([]byte, error) {
	data, err := json.Marshal(item)
	if err != nil {
		return nil, fmt.Errorf("marshal history item: %w", err)
	}

	compactData, err := compactJSON(data)
	if err != nil {
		return nil, fmt.Errorf("compact history item json: %w", err)
	}

	return compactData, nil
}

func unmarshalHistoryItemJSON(data []byte) (openai.ChatCompletionMessageParamUnion, error) {
	if !json.Valid(data) {
		return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("invalid history item json")
	}

	var item openai.ChatCompletionMessageParamUnion
	if err := json.Unmarshal(data, &item); err != nil {
		return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("unmarshal history item: %w", err)
	}

	return item, nil
}

func compactJSON(data []byte) ([]byte, error) {
	var compacted bytes.Buffer
	if err := json.Compact(&compacted, data); err != nil {
		return nil, err
	}
	return compacted.Bytes(), nil
}

// main runs an interactive DeepSeek chatbot session on stdin/stdout.
func main() {
	historyFile := flag.String("history-file", "", "load and append conversation history from a JSONL file")
	flag.Parse()

	apiKey := strings.TrimSpace(os.Getenv("DEEPSEEK_API_KEY"))
	baseURL := strings.TrimSpace(os.Getenv("DEEPSEEK_BASE_URL"))
	model := strings.TrimSpace(os.Getenv("DEEPSEEK_MODEL"))
	if baseURL == "" {
		baseURL = defaultBaseURL
	} else {
		fmt.Printf("Using custom DeepSeek API base URL: %s\n", baseURL)
	}
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "DEEPSEEK_API_KEY is not set")
		os.Exit(1)
	}
	if model == "" {
		model = defaultModel
	}

	debugLogFile, debugLogPath, err := openDebugLogFile()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to open DeepSeek debug log file: %v\n", err)
		os.Exit(1)
	}
	defer func() {
		if err := debugLogFile.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "failed to close DeepSeek debug log file: %v\n", err)
		}
	}()

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
		option.WithHeader("originator", "codex_cli_rs"),
		option.WithHeader("User-Agent", "codex_cli_rs/0.117.0 (Ubuntu 24.4.0; x86_64) WindowsTerminal (codex-tui; 0.117.0)"),
		option.WithDebugLog(log.New(&redactingLogWriter{file: debugLogFile}, "", log.LstdFlags|log.Lmicroseconds|log.LUTC)),
	)
	toolRegistry := tools.NewRegistry()
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, scannerInitialBytes), scannerMaxBytes)

	history, err := newChatHistory(*historyFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to initialize history: %v\n", err)
		os.Exit(1)
	}
	defer func() {
		if err := history.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "failed to close history file: %v\n", err)
		}
	}()

	if history.Len() == 0 {
		if err := history.Append(openai.SystemMessage(systemPrompt)); err != nil {
			fmt.Fprintf(os.Stderr, "failed to initialize history: %v\n", err)
			os.Exit(1)
		}
	}

	fmt.Printf("Simple DeepSeek chatbot started with model %s\n", model)
	fmt.Printf("DeepSeek debug log: %s\n", debugLogPath)
	if history.path != "" {
		fmt.Printf("History file: %s\n", history.path)
		if history.LoadedLen() > 0 {
			fmt.Printf("Loaded %d history items from %s\n", history.LoadedLen(), history.path)
		}
	}
	fmt.Println("Type a message and press Enter. Type 'exit' or 'quit' to stop.")

	ctx := context.Background()

	for {
		fmt.Print("You: ")
		if !scanner.Scan() {
			if err := scanner.Err(); err != nil {
				fmt.Fprintf(os.Stderr, "failed to read input: %v\n", err)
				os.Exit(1)
			}
			fmt.Println()
			return
		}

		userInput := strings.TrimSpace(scanner.Text())
		if userInput == "" {
			continue
		}

		switch strings.ToLower(userInput) {
		case "exit", "quit":
			fmt.Println("Bye.")
			return
		}

		if err := history.Append(openai.UserMessage(userInput)); err != nil {
			fmt.Fprintf(os.Stderr, "failed to append user input to history: %v\n\n", err)
			continue
		}

		if err := runMessage(ctx, client, model, history, toolRegistry); err != nil {
			fmt.Fprintf(os.Stderr, "%s\n\n", formatError(err))
			continue
		}
	}
}

// runMessage executes one committed user message, including follow-up tool-call rounds.
func runMessage(ctx context.Context, client openai.Client, model string, history *chatHistory, toolRegistry *tools.Registry) error {
	for range maxToolRounds {
		streamed, err := streamCompletion(ctx, client, model, history.Items(), toolRegistry.ChatDefinitions())
		if err != nil {
			return err
		}

		assistantMessage, toolCalls := buildAssistantReplay(streamed.completion)
		if err := history.Append(assistantMessage); err != nil {
			return fmt.Errorf("append assistant output to history: %w", err)
		}

		if len(toolCalls) == 0 {
			fmt.Print("\n\n")
			return nil
		}

		for _, call := range toolCalls {
			fmt.Printf("\nTool: %s\n", call.Function.Name)
			if err := history.Append(toolRegistry.ExecuteChat(ctx, call)); err != nil {
				return fmt.Errorf("append tool output to history: %w", err)
			}
		}
		fmt.Println()
	}

	return fmt.Errorf("tool call round limit exceeded (%d)", maxToolRounds)
}

// streamCompletion streams a single chat completion and captures the final payload.
func streamCompletion(ctx context.Context, client openai.Client, model string, history []openai.ChatCompletionMessageParamUnion, availableTools []openai.ChatCompletionToolUnionParam) (streamedCompletion, error) {
	params := openai.ChatCompletionNewParams{
		Model:    model,
		Messages: history,
		Tools:    availableTools,
	}
	params.SetExtraFields(map[string]any{
		"thinking": map[string]any{
			"type": "enabled",
		},
	})

	stream := client.Chat.Completions.NewStreaming(
		ctx,
		params,
	)

	var (
		acc    openai.ChatCompletionAccumulator
		result streamedCompletion
	)

	printReasoningHeader := true
	printOutputHeader := true

	for stream.Next() {
		chunk := stream.Current()
		if !acc.AddChunk(chunk) {
			return result, errors.New("chat completion chunk accumulation failed")
		}
		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]
		if reasoning := choice.Delta.ReasoningContent; reasoning != "" {
			if printReasoningHeader {
				fmt.Print("\nReasoning: ")
				printReasoningHeader = false
			}
			fmt.Print(reasoning)
		}

		if choice.Delta.Content != "" {
			if printOutputHeader {
				fmt.Print("\nOutput: ")
				printOutputHeader = false
			}
			fmt.Print(choice.Delta.Content)
		}

		if choice.Delta.Refusal != "" {
			if printOutputHeader {
				fmt.Print("\nOutput: ")
				printOutputHeader = false
			}
			fmt.Print(choice.Delta.Refusal)
		}
	}
	result.completion = acc.ChatCompletion

	err := stream.Err()
	closeErr := stream.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		return result, err
	}
	if len(result.completion.Choices) == 0 {
		return result, errors.New("chat completion stream ended without choices")
	}

	return result, nil
}

// formatError renders API and transport errors for terminal output.
func formatError(err error) string {
	var apiErr *openai.Error
	if errors.As(err, &apiErr) && apiErr.Message != "" {
		return fmt.Sprintf("DeepSeek API error (%d): %s", apiErr.StatusCode, apiErr.Message)
	}
	return fmt.Sprintf("request failed: %v", err)
}

func buildAssistantReplay(completion openai.ChatCompletion) (openai.ChatCompletionMessageParamUnion, []openai.ChatCompletionMessageFunctionToolCall) {
	if len(completion.Choices) == 0 {
		return openai.ChatCompletionMessageParamUnion{}, nil
	}

	message := completion.Choices[0].Message
	toolCalls := extractFunctionCalls(message)

	return message.ToParam(), toolCalls
}

func extractFunctionCalls(message openai.ChatCompletionMessage) []openai.ChatCompletionMessageFunctionToolCall {
	var calls []openai.ChatCompletionMessageFunctionToolCall
	for _, toolCall := range message.ToolCalls {
		// ChatCompletionAccumulator populates the inline fields on ToolCalls but not the raw JSON
		// backing needed by AsFunction(), so build the function call from the accumulated fields.
		if (toolCall.Type == "function" || toolCall.Type == "") && toolCall.ID != "" && toolCall.Function.Name != "" {
			calls = append(calls, openai.ChatCompletionMessageFunctionToolCall{
				ID: toolCall.ID,
				Function: openai.ChatCompletionMessageFunctionToolCallFunction{
					Name:      toolCall.Function.Name,
					Arguments: toolCall.Function.Arguments,
				},
			})
		}
	}

	return calls
}

func openDebugLogFile() (*os.File, string, error) {
	timestamp := time.Now().Format("20060102-150405.000000000")
	path := filepath.Join("debug_log", fmt.Sprintf("chatdeepseek-%s.log", timestamp))
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, "", err
	}
	return file, path, nil
}
