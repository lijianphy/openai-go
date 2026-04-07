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
	"slices"
	"strings"
	"time"

	"github.com/openai/openai-go/examples/tools"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

const (
	defaultBaseUrl = "https://api.openai.com/v1"
	defaultModel   = openai.ChatModelGPT5_4
	systemPrompt   = "You are a helpful chatbot. Keep replies concise and clear. Use available tools when they help answer the user."
	maxToolRounds  = 8
)

type streamedResponse struct {
	answer          string
	printedAnything bool
	response        responses.Response
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

// main runs an interactive OpenAI chatbot session on stdin/stdout.
func main() {
	apiKey := strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	baseUrl := strings.TrimSpace(os.Getenv("OPENAI_BASE_URL"))
	model := strings.TrimSpace(os.Getenv("OPENAI_MODEL"))
	if baseUrl == "" {
		baseUrl = defaultBaseUrl
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
		option.WithDebugLog(log.New(&redactingLogWriter{file: debugLogFile}, "", log.LstdFlags|log.Lmicroseconds|log.LUTC)),
	)
	toolRegistry := tools.NewRegistry()
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	history := responses.ResponseInputParam{
		responses.ResponseInputItemParamOfMessage(systemPrompt, responses.EasyInputMessageRoleSystem),
	}

	fmt.Printf("Simple OpenAI chatbot started with model %s\n", model)
	fmt.Printf("OpenAI debug log: %s\n", debugLogPath)
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

		history = append(history, responses.ResponseInputItemParamOfMessage(userInput, responses.EasyInputMessageRoleUser))

		nextHistory, err := runMessage(ctx, client, model, history, toolRegistry)
		history = nextHistory
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s\n\n", formatError(err))
			continue
		}
	}
}

// runMessage executes one committed user message, including any follow-up tool-call rounds.
func runMessage(ctx context.Context, client openai.Client, model string, history responses.ResponseInputParam, toolRegistry *tools.Registry) (responses.ResponseInputParam, error) {
	var combinedAnswer strings.Builder

	for range maxToolRounds {
		streamed, err := streamResponse(ctx, client, model, history, toolRegistry.Definitions())
		if err != nil {
			if streamed.printedAnything {
				fmt.Println()
			}
			return history, err
		}

		history = appendReplayItems(history, streamed.response)
		combinedAnswer.WriteString(streamed.answer)

		toolCalls := extractFunctionCalls(streamed.response)
		if len(toolCalls) == 0 {
			if strings.TrimSpace(combinedAnswer.String()) == "" {
				if streamed.printedAnything {
					fmt.Println()
				}
				fmt.Print("Output: [no text returned]")
			}
			fmt.Print("\n\n")
			return history, nil
		}

		if streamed.printedAnything {
			fmt.Println()
		}
		for _, call := range toolCalls {
			fmt.Printf("Tool: %s\n", call.Name)
			history = append(history, toolRegistry.Execute(ctx, call))
		}
		fmt.Println()
	}

	return history, fmt.Errorf("tool call round limit exceeded (%d)", maxToolRounds)
}

// streamResponse streams a single model response and captures the final payload.
func streamResponse(ctx context.Context, client openai.Client, model string, history responses.ResponseInputParam, availableTools []responses.ToolUnionParam) (streamedResponse, error) {
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

	var result streamedResponse
	printReasoningHeader := true
	printOutputHeader := true
	outputItems := make(map[int64]responses.ResponseOutputItemUnion)
	outputIndexes := make([]int64, 0)

	for stream.Next() {
		event := stream.Current()

		switch event.Type {
		case "response.reasoning_summary_text.delta":
			if printReasoningHeader {
				fmt.Print("Reasoning: ")
				printReasoningHeader = false
				result.printedAnything = true
			}
			fmt.Print(event.Delta)
			result.printedAnything = true
		case "response.reasoning_summary_text.done":
			if printReasoningHeader && event.Text != "" {
				fmt.Print("Reasoning: ")
				printReasoningHeader = false
				result.printedAnything = true
				fmt.Print(event.Text)
				result.printedAnything = true
			}
		case "response.output_text.delta", "response.refusal.delta":
			if printOutputHeader {
				if !printReasoningHeader {
					fmt.Print("\n")
				}
				fmt.Print("Output: ")
				printOutputHeader = false
				result.printedAnything = true
			}
			fmt.Print(event.Delta)
			result.answer += event.Delta
			result.printedAnything = true
		case "response.output_text.done":
			if result.answer == "" && event.Text != "" {
				if printOutputHeader {
					if !printReasoningHeader {
						fmt.Print("\n")
					}
					fmt.Print("Output: ")
					printOutputHeader = false
					result.printedAnything = true
				}
				fmt.Print(event.Text)
				result.answer += event.Text
				result.printedAnything = true
			}
		case "response.refusal.done":
			if result.answer == "" && event.Refusal != "" {
				if printOutputHeader {
					if !printReasoningHeader {
						fmt.Print("\n")
					}
					fmt.Print("Output: ")
					printOutputHeader = false
					result.printedAnything = true
				}
				fmt.Print(event.Refusal)
				result.answer += event.Refusal
				result.printedAnything = true
			}
		case "response.output_item.done":
			if _, ok := outputItems[event.OutputIndex]; !ok {
				outputIndexes = append(outputIndexes, event.OutputIndex)
			}
			outputItems[event.OutputIndex] = event.Item
		case "response.completed", "response.incomplete", "response.failed":
			result.response = event.Response
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
	if result.response.ID == "" {
		return result, errors.New("response stream ended without a completed response")
	}
	if result.response.Status == responses.ResponseStatusFailed {
		if result.response.Error.Message != "" {
			return result, errors.New(result.response.Error.Message)
		}
		return result, errors.New("response failed")
	}
	if result.response.Status == responses.ResponseStatusIncomplete {
		if result.response.IncompleteDetails.Reason != "" {
			return result, fmt.Errorf("response incomplete: %s", result.response.IncompleteDetails.Reason)
		}
		return result, errors.New("response incomplete")
	}
	if len(outputItems) > 0 {
		result.response.Output = orderedOutputItems(outputItems, outputIndexes)
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
func appendReplayItems(history responses.ResponseInputParam, resp responses.Response) responses.ResponseInputParam {
	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			message := item.AsMessage()
			messageParam := message.ToParam()
			history = append(history, responses.ResponseInputItemUnionParam{
				OfOutputMessage: &messageParam,
			})
		case "reasoning":
			reasoning := item.AsReasoning()
			reasoningParam := reasoning.ToParam()
			history = append(history, responses.ResponseInputItemUnionParam{
				OfReasoning: &reasoningParam,
			})
		case "function_call":
			functionCall := item.AsFunctionCall()
			functionCallParam := functionCall.ToParam()
			history = append(history, responses.ResponseInputItemUnionParam{
				OfFunctionCall: &functionCallParam,
			})
		}
	}

	return history
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
