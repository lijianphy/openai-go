# Repository Guidelines

## Project Structure & Module Organization
The root package contains the primary `openai` client and most generated resource files such as `chatcompletion.go` and `vectorstore.go`. Focused API areas live in subpackages like `responses/`, `realtime/`, `conversations/`, `webhooks/`, `azure/`, and `option/`. Shared helpers sit in `packages/` and `internal/`. Tests live beside implementation files as `*_test.go`. `examples/` is a separate Go module with runnable samples; it is safe to add or edit examples there. Most SDK files are generated from the OpenAPI spec by Stainless, so preserve generated headers and keep manual edits narrowly scoped.

## Build, Test, and Development Commands
Use `./scripts/bootstrap` to install or tidy dependencies for the root module. Run `./scripts/lint` before opening a PR; it builds the SDK, checks test compilation, and builds the `examples/` module. Use `./scripts/test` for the full suite; it starts `./scripts/mock --daemon` automatically unless `TEST_API_BASE_URL` is already set. Format with `./scripts/format`, which runs `gofmt -s -w .`. Verify module files with `./scripts/check-go-mod`.

## Coding Style & Naming Conventions
Target Go 1.22+ and follow standard Go formatting with tabs via `gofmt`. Keep package names lowercase, exported identifiers in `CamelCase`, and test files named `*_test.go`. Match existing resource-oriented filenames when adding new surfaces, for example `responses/inputitem.go` or `webhooks/webhook.go`. Prefer small, local changes over broad rewrites in generated files, and place handwritten helpers in stable non-generated areas when possible.

## Testing Guidelines
Write table-driven Go tests where that matches nearby files, and keep tests next to the code they cover. Run `./scripts/test` for behavior changes and `go test ./... -run '^$'` through `./scripts/lint` for compile-only verification. If your change affects examples or module dependencies, also run `./scripts/check-go-mod` so both `go.mod` files stay tidy.

## Commit & Pull Request Guidelines
Recent history favors short imperative subjects, often in Conventional Commit form such as `fix(api): ...`, `feat(api): ...`, or `chore(tests): ...`. Keep commits focused and lower-case. PRs should summarize the user-visible change, note any generated-code or spec impact, and list the validation you ran, typically `./scripts/lint`, `./scripts/test`, and `./scripts/check-go-mod`. Include example or docs updates when public behavior changes. Breaking API changes receive extra scrutiny on PRs targeting `main` or `next`.
