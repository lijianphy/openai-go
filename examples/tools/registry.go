package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
)

type Handler func(ctx context.Context, arguments string) string

type Tool struct {
	Name           string
	Definition     responses.ToolUnionParam
	ChatDefinition openai.ChatCompletionToolUnionParam
	Handler        Handler
}

type Registry struct {
	definitions     []responses.ToolUnionParam
	chatDefinitions []openai.ChatCompletionToolUnionParam
	handlers        map[string]Handler
}

func NewRegistry() *Registry {
	registry := &Registry{
		handlers: make(map[string]Handler),
	}
	registry.Register(newBashTool())
	return registry
}

func (r *Registry) Register(tool Tool) {
	if _, exists := r.handlers[tool.Name]; exists {
		panic(fmt.Sprintf("tool %q is already registered", tool.Name))
	}

	r.definitions = append(r.definitions, tool.Definition)
	if hasChatDefinition(tool.ChatDefinition) {
		r.chatDefinitions = append(r.chatDefinitions, tool.ChatDefinition)
	}
	r.handlers[tool.Name] = tool.Handler
}

func (r *Registry) Definitions() []responses.ToolUnionParam {
	return append([]responses.ToolUnionParam(nil), r.definitions...)
}

func (r *Registry) ChatDefinitions() []openai.ChatCompletionToolUnionParam {
	return append([]openai.ChatCompletionToolUnionParam(nil), r.chatDefinitions...)
}

func (r *Registry) Execute(ctx context.Context, call responses.ResponseFunctionToolCall) responses.ResponseInputItemUnionParam {
	handler, ok := r.handlers[call.Name]
	if !ok {
		return responses.ResponseInputItemParamOfFunctionCallOutput(call.CallID, unknownToolOutput(call.Name))
	}

	return responses.ResponseInputItemParamOfFunctionCallOutput(call.CallID, handler(ctx, call.Arguments))
}

func (r *Registry) ExecuteChat(ctx context.Context, call openai.ChatCompletionMessageFunctionToolCall) openai.ChatCompletionMessageParamUnion {
	handler, ok := r.handlers[call.Function.Name]
	if !ok {
		return openai.ToolMessage(unknownToolOutput(call.Function.Name), call.ID)
	}

	return openai.ToolMessage(handler(ctx, call.Function.Arguments), call.ID)
}

func hasChatDefinition(definition openai.ChatCompletionToolUnionParam) bool {
	return definition.OfFunction != nil || definition.OfCustom != nil
}

func unknownToolOutput(name string) string {
	return marshalToolOutput(map[string]any{
		"exit_code": -1,
		"error":     fmt.Sprintf("unknown tool %q", name),
	})
}

func marshalToolOutput(v any) string {
	data, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf(`{"exit_code":-1,"error":"failed to encode tool output: %s"}`, err)
	}

	return string(data)
}
