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
	"slices"
	"strings"
	"time"

	"github.com/openai/openai-go/examples/tools"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

const (
	defaultBaseURL      = "https://api.openai.com/v1"
	defaultModel        = openai.ChatModelGPT5_4
	systemPrompt        = "You are a helpful chatbot. Keep replies concise and clear. Use available tools when they help answer the user."
	maxToolRounds       = 8
	scannerInitialBytes = 64 * 1024
	scannerMaxBytes     = 1024 * 1024
	historyLineMaxBytes = 8 * 1024 * 1024
)

var authorizationHeaderPattern = regexp.MustCompile(`(?im)^(Authorization:\s*Bearer\s+)[^\r\n]+`)

type redactingLogWriter struct {
	file *os.File
}

type chatHistory struct {
	file       *os.File
	path       string
	items      responses.ResponseInputParam
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

func (h *chatHistory) Items() responses.ResponseInputParam {
	return append(responses.ResponseInputParam(nil), h.items...)
}

func (h *chatHistory) Append(item responses.ResponseInputItemUnionParam) error {
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

func loadHistoryFile(path string) (responses.ResponseInputParam, error) {
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

	var items responses.ResponseInputParam
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

func appendHistoryItemJSONL(file *os.File, item responses.ResponseInputItemUnionParam) error {
	line, err := marshalHistoryItemJSON(item)
	if err != nil {
		return err
	}

	if _, err := file.Write(append(line, '\n')); err != nil {
		return fmt.Errorf("write history file %q: %w", file.Name(), err)
	}

	return nil
}

func marshalHistoryItemJSON(item responses.ResponseInputItemUnionParam) ([]byte, error) {
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

func unmarshalHistoryItemJSON(data []byte) (responses.ResponseInputItemUnionParam, error) {
	if !json.Valid(data) {
		return responses.ResponseInputItemUnionParam{}, fmt.Errorf("invalid history item json")
	}

	return param.Override[responses.ResponseInputItemUnionParam](json.RawMessage(data)), nil
}

func compactJSON(data []byte) ([]byte, error) {
	var compacted bytes.Buffer
	if err := json.Compact(&compacted, data); err != nil {
		return nil, err
	}
	return compacted.Bytes(), nil
}

// main runs an interactive OpenAI chatbot session on stdin/stdout.
func main() {
	historyFile := flag.String("history-file", "", "load and append conversation history from a JSONL file")
	flag.Parse()

	apiKey := strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	baseUrl := strings.TrimSpace(os.Getenv("OPENAI_BASE_URL"))
	model := strings.TrimSpace(os.Getenv("OPENAI_MODEL"))
	if baseUrl == "" {
		baseUrl = defaultBaseURL
	} else {
		fmt.Printf("Using custom OpenAI API base URL: %s\n", baseUrl)
	}
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "OPENAI_API_KEY is not set")
		os.Exit(1)
	}
	if model == "" {
		model = defaultModel
	}

	debugLogFile, debugLogPath, err := openDebugLogFile()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to open OpenAI debug log file: %v\n", err)
		os.Exit(1)
	}
	defer func() {
		if err := debugLogFile.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "failed to close OpenAI debug log file: %v\n", err)
		}
	}()

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseUrl),
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
		if err := history.Append(responses.ResponseInputItemParamOfMessage(systemPrompt, responses.EasyInputMessageRoleSystem)); err != nil {
			fmt.Fprintf(os.Stderr, "failed to initialize history: %v\n", err)
			os.Exit(1)
		}
	}

	fmt.Printf("Simple OpenAI chatbot started with model %s\n", model)
	fmt.Printf("OpenAI debug log: %s\n", debugLogPath)
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

		if err := history.Append(responses.ResponseInputItemParamOfMessage(userInput, responses.EasyInputMessageRoleUser)); err != nil {
			fmt.Fprintf(os.Stderr, "failed to append user input to history: %v\n\n", err)
			continue
		}

		if err := runMessage(ctx, client, model, history, toolRegistry); err != nil {
			fmt.Fprintf(os.Stderr, "%s\n\n", formatError(err))
			continue
		}
	}
}

// runMessage executes one committed user message, including any follow-up tool-call rounds.
func runMessage(ctx context.Context, client openai.Client, model string, history *chatHistory, toolRegistry *tools.Registry) error {
	for range maxToolRounds {
		response, err := streamResponse(ctx, client, model, history.Items(), toolRegistry.Definitions())
		if err != nil {
			fmt.Println()
			return err
		}

		if err := appendReplayItems(history, response); err != nil {
			return err
		}

		toolCalls := extractFunctionCalls(response)
		if len(toolCalls) == 0 {
			fmt.Print("\n\n")
			return nil
		}

		for _, call := range toolCalls {
			fmt.Printf("\nTool: %s\n", call.Name)
			if err := history.Append(toolRegistry.Execute(ctx, call)); err != nil {
				return err
			}
		}
		fmt.Println()
	}

	return fmt.Errorf("tool call round limit exceeded (%d)", maxToolRounds)
}

// streamResponse streams a single model response and captures the final payload.
func streamResponse(ctx context.Context, client openai.Client, model string, history responses.ResponseInputParam, availableTools []responses.ToolUnionParam) (responses.Response, error) {
	stream := client.Responses.NewStreaming(ctx, responses.ResponseNewParams{
		Model: model,
		Include: []responses.ResponseIncludable{
			responses.ResponseIncludableReasoningEncryptedContent,
		},
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: history,
		},
		Store: openai.Bool(false),
		Reasoning: openai.ReasoningParam{
			Effort:  shared.ReasoningEffortXhigh,
			Summary: openai.ReasoningSummaryDetailed,
		},
		Tools: availableTools,
	})

	var result responses.Response
	printReasoningHeader := true
	printOutputHeader := true
	outputItems := make(map[int64]responses.ResponseOutputItemUnion)
	outputIndexes := make([]int64, 0)

	for stream.Next() {
		event := stream.Current()

		switch event.Type {
		case "response.reasoning_summary_text.delta":
			if printReasoningHeader {
				fmt.Print("\nReasoning: ")
				printReasoningHeader = false
			}
			fmt.Print(event.Delta)
		case "response.output_text.delta", "response.refusal.delta":
			if printOutputHeader {
				fmt.Print("\nOutput: ")
				printOutputHeader = false
			}
			fmt.Print(event.Delta)
		case "response.output_item.done":
			if _, ok := outputItems[event.OutputIndex]; !ok {
				outputIndexes = append(outputIndexes, event.OutputIndex)
			}
			outputItems[event.OutputIndex] = event.Item
		case "response.completed", "response.incomplete", "response.failed":
			result = event.Response
		}
	}

	err := stream.Err()
	closeErr := stream.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		return result, err
	}
	if result.ID == "" {
		return result, errors.New("response stream ended without a completed response")
	}
	if result.Status == responses.ResponseStatusFailed {
		if result.Error.Message != "" {
			return result, errors.New(result.Error.Message)
		}
		return result, errors.New("response failed")
	}
	if result.Status == responses.ResponseStatusIncomplete {
		if result.IncompleteDetails.Reason != "" {
			return result, fmt.Errorf("response incomplete: %s", result.IncompleteDetails.Reason)
		}
		return result, errors.New("response incomplete")
	}
	if len(outputItems) > 0 {
		result.Output = orderedOutputItems(outputItems, outputIndexes)
	}

	return result, nil
}

func orderedOutputItems(items map[int64]responses.ResponseOutputItemUnion, indexes []int64) []responses.ResponseOutputItemUnion {
	sorted := append([]int64(nil), indexes...)
	slices.Sort(sorted)

	output := make([]responses.ResponseOutputItemUnion, 0, len(sorted))
	for _, index := range sorted {
		output = append(output, items[index])
	}

	return output
}

// formatError renders API and transport errors for terminal output.
func formatError(err error) string {
	var apiErr *openai.Error
	if errors.As(err, &apiErr) && apiErr.Message != "" {
		return fmt.Sprintf("OpenAI API error (%d): %s", apiErr.StatusCode, apiErr.Message)
	}
	return fmt.Sprintf("request failed: %v", err)
}

// appendReplayItems converts response items into input items for the next round.
func appendReplayItems(history *chatHistory, resp responses.Response) error {
	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			message := item.AsMessage()
			messageParam := message.ToParam()
			if err := history.Append(responses.ResponseInputItemUnionParam{
				OfOutputMessage: &messageParam,
			}); err != nil {
				return fmt.Errorf("append output message to history: %w", err)
			}
		case "reasoning":
			reasoning := item.AsReasoning()
			reasoningParam := reasoning.ToParam()
			if err := history.Append(responses.ResponseInputItemUnionParam{
				OfReasoning: &reasoningParam,
			}); err != nil {
				return fmt.Errorf("append reasoning item to history: %w", err)
			}
		case "function_call":
			functionCall := item.AsFunctionCall()
			functionCallParam := functionCall.ToParam()
			if err := history.Append(responses.ResponseInputItemUnionParam{
				OfFunctionCall: &functionCallParam,
			}); err != nil {
				return fmt.Errorf("append function call to history: %w", err)
			}
		}
	}

	return nil
}

// extractFunctionCalls collects function-call items from a model response.
func extractFunctionCalls(resp responses.Response) []responses.ResponseFunctionToolCall {
	var calls []responses.ResponseFunctionToolCall
	for _, item := range resp.Output {
		if item.Type == "function_call" {
			calls = append(calls, item.AsFunctionCall())
		}
	}

	return calls
}

func openDebugLogFile() (*os.File, string, error) {
	timestamp := time.Now().Format("20060102-150405.000000000")
	path := filepath.Join("debug_log", fmt.Sprintf("chatopenai-%s.log", timestamp))
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, "", err
	}
	return file, path, nil
}
