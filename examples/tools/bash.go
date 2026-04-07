package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

const (
	BashToolName      = "bash"
	bashOutputLimit   = 32 * 1024
	bashTimeout       = 30 * time.Second
	bashToolDesc      = "Executes bash script in a fresh bash shell."
	bashScriptArgDesc = "The bash script to execute in the bash shell. The script may be multiline."
)

type bashToolArgs struct {
	Script string `json:"script"`
}

type bashToolResult struct {
	ExitCode  int    `json:"exit_code"`
	Stdout    string `json:"stdout,omitempty"`
	Stderr    string `json:"stderr,omitempty"`
	Error     string `json:"error,omitempty"`
	Truncated bool   `json:"truncated,omitempty"`
}

func newBashTool() Tool {
	parameters := map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"properties": map[string]any{
			"script": map[string]any{
				"type":        "string",
				"description": bashScriptArgDesc,
			},
		},
		"required": []string{"script"},
	}

	return Tool{
		Name: BashToolName,
		Definition: responses.ToolUnionParam{
			OfFunction: &responses.FunctionToolParam{
				Name:        BashToolName,
				Description: openai.String(bashToolDesc),
				Parameters:  parameters,
				Strict:      openai.Bool(true),
			},
		},
		ChatDefinition: openai.ChatCompletionToolUnionParam{
			OfFunction: &openai.ChatCompletionFunctionToolParam{
				Function: shared.FunctionDefinitionParam{
					Name:        BashToolName,
					Description: openai.String(bashToolDesc),
					Parameters:  shared.FunctionParameters(parameters),
					Strict:      openai.Bool(true),
				},
			},
		},
		Handler: runBashTool,
	}
}

func runBashTool(ctx context.Context, arguments string) string {
	var args bashToolArgs
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return finalizeBashToolResult(bashToolResult{
			ExitCode: -1,
			Error:    fmt.Sprintf("invalid arguments: %v", err),
		})
	}

	if strings.TrimSpace(args.Script) == "" {
		return finalizeBashToolResult(bashToolResult{
			ExitCode: -1,
			Error:    "script is required",
		})
	}

	fmt.Fprintf(os.Stderr, "bash input:\n%s\n", args.Script)

	runCtx, cancel := context.WithTimeout(ctx, bashTimeout)
	defer cancel()

	cmd := exec.CommandContext(runCtx, "bash", "-c", args.Script)

	var stdout limitedBuffer
	var stderr limitedBuffer
	stdout.limit = bashOutputLimit
	stderr.limit = bashOutputLimit
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	result := bashToolResult{}
	result.Stdout = stdout.String()
	result.Stderr = stderr.String()
	result.Truncated = stdout.truncated || stderr.truncated

	if errors.Is(runCtx.Err(), context.DeadlineExceeded) {
		result.ExitCode = -1
		result.Error = fmt.Sprintf("script timed out after %s", bashTimeout)
		return finalizeBashToolResult(result)
	}

	if err == nil {
		result.ExitCode = 0
		return finalizeBashToolResult(result)
	}

	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		result.ExitCode = exitErr.ExitCode()
		return finalizeBashToolResult(result)
	}

	result.ExitCode = -1
	result.Error = err.Error()
	return finalizeBashToolResult(result)
}

func finalizeBashToolResult(result bashToolResult) string {
	output := marshalToolOutput(result)
	fmt.Fprintf(os.Stderr, "bash output: %s\n", output)
	return output
}

type limitedBuffer struct {
	buf       bytes.Buffer
	limit     int
	truncated bool
}

func (b *limitedBuffer) Write(p []byte) (int, error) {
	if b.limit <= 0 {
		b.truncated = true
		return len(p), nil
	}

	remaining := b.limit - b.buf.Len()
	if remaining <= 0 {
		b.truncated = true
		return len(p), nil
	}

	if len(p) > remaining {
		p = p[:remaining]
		b.truncated = true
	}

	_, _ = b.buf.Write(p)
	return len(p), nil
}

func (b *limitedBuffer) String() string {
	return b.buf.String()
}
