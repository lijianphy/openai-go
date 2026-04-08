package main

import (
	"bufio"
	"context"
	"errors"
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
	defaultBaseURL = "https://api.deepseek.com/v1"
	defaultModel   = "deepseek-reasoner"
	systemPrompt   = "You are a helpful chatbot. Keep replies concise and clear. Use available tools when they help answer the user."
	maxToolRounds  = 8
)

type streamedCompletion struct {
	completion openai.ChatCompletion
}

var authorizationHeaderPattern = regexp.MustCompile(`(?im)^(Authorization:\s*Bearer\s+)[^\r\n]+`)

type redactingLogWriter struct {
	file *os.File
}

func (w *redactingLogWriter) Write(p []byte) (int, error) {
	redacted := authorizationHeaderPattern.ReplaceAll(p, []byte("${1}[REDACTED]"))
	if _, err := w.file.Write(redacted); err != nil {
		return 0, err
	}

	return len(p), nil
}

// main runs an interactive DeepSeek chatbot session on stdin/stdout.
func main() {
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
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	history := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(systemPrompt),
	}

	fmt.Printf("Simple DeepSeek chatbot started with model %s\n", model)
	fmt.Printf("DeepSeek debug log: %s\n", debugLogPath)
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

		history = append(history, openai.UserMessage(userInput))

		nextHistory, err := runMessage(ctx, client, model, history, toolRegistry)
		history = nextHistory
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s\n\n", formatError(err))
			continue
		}
	}
}

// runMessage executes one committed user message, including follow-up tool-call rounds.
func runMessage(ctx context.Context, client openai.Client, model string, history []openai.ChatCompletionMessageParamUnion, toolRegistry *tools.Registry) ([]openai.ChatCompletionMessageParamUnion, error) {
	for range maxToolRounds {
		streamed, err := streamCompletion(ctx, client, model, history, toolRegistry.ChatDefinitions())
		if err != nil {
			return history, err
		}

		assistantMessage, toolCalls := buildAssistantReplay(streamed.completion)
		history = append(history, assistantMessage)

		if len(toolCalls) == 0 {
			fmt.Print("\n\n")
			return history, nil
		}

		for _, call := range toolCalls {
			fmt.Printf("\nTool: %s\n", call.Function.Name)
			history = append(history, toolRegistry.ExecuteChat(ctx, call))
		}
		fmt.Println()
	}

	return history, fmt.Errorf("tool call round limit exceeded (%d)", maxToolRounds)
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
