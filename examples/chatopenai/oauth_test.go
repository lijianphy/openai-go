package main

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestOAuthTokenSourceBrowserLoginPersistsSession(t *testing.T) {
	t.Parallel()

	idToken := testJWT(t, map[string]any{
		"https://api.openai.com/auth": map[string]any{
			"chatgpt_account_id": "acct_321",
		},
	})

	var authorizeCalls atomic.Int32
	var tokenCalls atomic.Int32
	var authorizeState syncValue[string]
	var authorizeRedirectURI syncValue[string]
	var authorizeChallenge syncValue[string]
	const allowedWorkspaceID = "acct_321"

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/oauth/authorize":
			authorizeCalls.Add(1)

			query := r.URL.Query()
			if got := query.Get("response_type"); got != "code" {
				t.Fatalf("response_type = %q, want %q", got, "code")
			}
			if got := query.Get("client_id"); got != "client-id" {
				t.Fatalf("client_id = %q, want %q", got, "client-id")
			}
			if got := query.Get("code_challenge_method"); got != "S256" {
				t.Fatalf("code_challenge_method = %q, want %q", got, "S256")
			}
			if got := query.Get("originator"); got != oauthLoginOriginator {
				t.Fatalf("originator = %q, want %q", got, oauthLoginOriginator)
			}
			if got := query.Get("scope"); got != oauthAuthorizeScopes {
				t.Fatalf("scope = %q, want %q", got, oauthAuthorizeScopes)
			}
			if got := query.Get("allowed_workspace_id"); got != allowedWorkspaceID {
				t.Fatalf("allowed_workspace_id = %q, want %q", got, allowedWorkspaceID)
			}

			redirectURI := query.Get("redirect_uri")
			state := query.Get("state")
			challenge := query.Get("code_challenge")
			if redirectURI == "" {
				t.Fatalf("redirect_uri is empty")
			}
			if state == "" {
				t.Fatalf("state is empty")
			}
			if challenge == "" {
				t.Fatalf("code_challenge is empty")
			}

			authorizeRedirectURI.Store(redirectURI)
			authorizeState.Store(state)
			authorizeChallenge.Store(challenge)

			callbackURI, err := url.Parse(redirectURI)
			if err != nil {
				t.Fatalf("url.Parse() error = %v", err)
			}
			callbackURI.Host = strings.Replace(callbackURI.Host, "localhost:", "127.0.0.1:", 1)

			callbackQuery := callbackURI.Query()
			callbackQuery.Set("code", "browser-code-321")
			callbackQuery.Set("state", state)
			callbackURI.RawQuery = callbackQuery.Encode()

			http.Redirect(w, r, callbackURI.String(), http.StatusFound)
		case "/oauth/token":
			tokenCalls.Add(1)

			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatalf("io.ReadAll() error = %v", err)
			}

			values, err := url.ParseQuery(string(body))
			if err != nil {
				t.Fatalf("url.ParseQuery() error = %v", err)
			}

			if got := values.Get("grant_type"); got != "authorization_code" {
				t.Fatalf("grant_type = %q, want %q", got, "authorization_code")
			}
			if got := values.Get("code"); got != "browser-code-321" {
				t.Fatalf("code = %q, want %q", got, "browser-code-321")
			}
			if got := values.Get("redirect_uri"); got != authorizeRedirectURI.Load() {
				t.Fatalf("redirect_uri = %q, want %q", got, authorizeRedirectURI.Load())
			}

			codeVerifier := values.Get("code_verifier")
			if codeVerifier == "" {
				t.Fatalf("code_verifier is empty")
			}
			sum := sha256.Sum256([]byte(codeVerifier))
			challenge := base64.RawURLEncoding.EncodeToString(sum[:])
			if challenge != authorizeChallenge.Load() {
				t.Fatalf("code challenge = %q, want %q", challenge, authorizeChallenge.Load())
			}

			writeJSON(t, w, http.StatusOK, map[string]any{
				"id_token":      idToken,
				"access_token":  "access-token-123",
				"refresh_token": "refresh-token-123",
			})
		default:
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
	}))
	defer server.Close()

	prompt := &lockedBuffer{}
	cfg := oauthConfig{
		issuer:             server.URL,
		clientID:           "client-id",
		refreshURL:         server.URL + "/oauth/token",
		allowedWorkspaceID: allowedWorkspaceID,
		codexHome:          t.TempDir(),
		httpClient:         server.Client(),
		prompt:             prompt,
	}

	source, err := newOAuthTokenSource(cfg)
	if err != nil {
		t.Fatalf("newOAuthTokenSource() error = %v", err)
	}

	type tokenResult struct {
		token string
		err   error
	}
	resultCh := make(chan tokenResult, 1)
	go func() {
		token, err := source.ensureRequestCredential(context.Background())
		resultCh <- tokenResult{token: token, err: err}
	}()

	authURL := waitForPromptURL(t, prompt, server.URL+"/oauth/authorize")
	resp, err := http.Get(authURL)
	if err != nil {
		t.Fatalf("http.Get() error = %v", err)
	}
	callbackBody, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		t.Fatalf("io.ReadAll() error = %v", err)
	}
	if !strings.Contains(string(callbackBody), "Authentication complete") {
		t.Fatalf("callback body = %q, want completion message", string(callbackBody))
	}

	select {
	case result := <-resultCh:
		if result.err != nil {
			t.Fatalf("ensureRequestCredential() error = %v", result.err)
		}
		if result.token != "access-token-123" {
			t.Fatalf("request credential = %q, want %q", result.token, "access-token-123")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for browser login to complete")
	}

	if got := authorizeCalls.Load(); got != 1 {
		t.Fatalf("authorize calls = %d, want 1", got)
	}
	if got := tokenCalls.Load(); got != 1 {
		t.Fatalf("token calls = %d, want 1", got)
	}
	if authorizeState.Load() == "" {
		t.Fatalf("authorize state was not captured")
	}

	rawAuth, err := os.ReadFile(filepath.Join(cfg.codexHome, "auth.json"))
	if err != nil {
		t.Fatalf("os.ReadFile() error = %v", err)
	}

	var auth oauthAuthFile
	if err := json.Unmarshal(rawAuth, &auth); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if auth.Tokens == nil {
		t.Fatalf("stored auth tokens are nil")
	}
	if auth.OpenAIAPIKey != "" {
		t.Fatalf("stored OpenAI API key = %q, want empty", auth.OpenAIAPIKey)
	}
	if auth.Tokens.AccessToken != "access-token-123" {
		t.Fatalf("stored access token = %q, want %q", auth.Tokens.AccessToken, "access-token-123")
	}
	if auth.Tokens.RefreshToken != "refresh-token-123" {
		t.Fatalf("stored refresh token = %q, want %q", auth.Tokens.RefreshToken, "refresh-token-123")
	}
	if auth.Tokens.IDToken != idToken {
		t.Fatalf("stored id token = %q, want %q", auth.Tokens.IDToken, idToken)
	}
	if auth.Tokens.AccountID != "acct_321" {
		t.Fatalf("stored account id = %q, want %q", auth.Tokens.AccountID, "acct_321")
	}
	if auth.LastRefresh == nil {
		t.Fatalf("stored last_refresh is nil")
	}
}

func TestOAuthMiddlewareRefreshesUnauthorized(t *testing.T) {
	t.Parallel()

	var refreshCalls atomic.Int32
	oldIDToken := testJWT(t, map[string]any{
		"https://api.openai.com/auth": map[string]any{
			"chatgpt_account_id": "acct_123",
		},
	})
	refreshedIDToken := testJWT(t, map[string]any{
		"https://api.openai.com/auth": map[string]any{
			"chatgpt_account_id": "acct_987",
		},
	})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/oauth/token" {
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Fatalf("refresh method = %s, want POST", r.Method)
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("io.ReadAll() error = %v", err)
		}

		if !strings.HasPrefix(r.Header.Get("Content-Type"), "application/json") {
			t.Fatalf("content-type = %q, want application/json", r.Header.Get("Content-Type"))
		}

		refreshCalls.Add(1)

		var refreshBody map[string]string
		if err := json.Unmarshal(body, &refreshBody); err != nil {
			t.Fatalf("json.Unmarshal() error = %v", err)
		}
		if refreshBody["grant_type"] != "refresh_token" {
			t.Fatalf("grant_type = %q, want %q", refreshBody["grant_type"], "refresh_token")
		}
		if refreshBody["refresh_token"] != "refresh-token" {
			t.Fatalf("refresh_token = %q, want %q", refreshBody["refresh_token"], "refresh-token")
		}

		writeJSON(t, w, http.StatusOK, map[string]any{
			"id_token":      refreshedIDToken,
			"access_token":  "new-token",
			"refresh_token": "new-refresh-token",
		})
	}))
	defer server.Close()

	cfg := oauthConfig{
		issuer:     server.URL,
		clientID:   "client-id",
		refreshURL: server.URL + "/oauth/token",
		codexHome:  t.TempDir(),
		httpClient: server.Client(),
		prompt:     io.Discard,
	}

	source, err := newOAuthTokenSource(cfg)
	if err != nil {
		t.Fatalf("newOAuthTokenSource() error = %v", err)
	}

	source.auth.Tokens = &oauthTokens{
		IDToken:      oldIDToken,
		AccessToken:  "old-token",
		RefreshToken: "refresh-token",
	}

	var requestCalls atomic.Int32
	middleware := source.middleware()
	req, err := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/responses", strings.NewReader(`{"input":"hello"}`))
	if err != nil {
		t.Fatalf("http.NewRequest() error = %v", err)
	}

	resp, err := middleware(req, func(request *http.Request) (*http.Response, error) {
		call := requestCalls.Add(1)
		switch call {
		case 1:
			if got := request.Header.Get("Authorization"); got != "Bearer old-token" {
				t.Fatalf("first authorization header = %q, want %q", got, "Bearer old-token")
			}
			if got := request.Header.Get("ChatGPT-Account-ID"); got != "acct_123" {
				t.Fatalf("first account header = %q, want %q", got, "acct_123")
			}
			return &http.Response{
				StatusCode: http.StatusUnauthorized,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(`{"error":"expired"}`)),
				Request:    request,
			}, nil
		case 2:
			if got := request.Header.Get("Authorization"); got != "Bearer new-token" {
				t.Fatalf("second authorization header = %q, want %q", got, "Bearer new-token")
			}
			if got := request.Header.Get("ChatGPT-Account-ID"); got != "acct_987" {
				t.Fatalf("second account header = %q, want %q", got, "acct_987")
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(`{"ok":true}`)),
				Request:    request,
			}, nil
		default:
			t.Fatalf("unexpected request attempt %d", call)
			return nil, nil
		}
	})
	if err != nil {
		t.Fatalf("middleware() error = %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("response status = %d, want %d", resp.StatusCode, http.StatusOK)
	}
	if got := requestCalls.Load(); got != 2 {
		t.Fatalf("request calls = %d, want 2", got)
	}
	if got := refreshCalls.Load(); got != 1 {
		t.Fatalf("refresh calls = %d, want 1", got)
	}

	rawAuth, err := os.ReadFile(filepath.Join(cfg.codexHome, "auth.json"))
	if err != nil {
		t.Fatalf("os.ReadFile() error = %v", err)
	}

	var auth oauthAuthFile
	if err := json.Unmarshal(rawAuth, &auth); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if auth.Tokens == nil {
		t.Fatalf("stored auth tokens are nil")
	}
	if auth.OpenAIAPIKey != "" {
		t.Fatalf("stored OpenAI API key = %q, want empty", auth.OpenAIAPIKey)
	}
	if auth.Tokens.AccessToken != "new-token" {
		t.Fatalf("stored access token = %q, want %q", auth.Tokens.AccessToken, "new-token")
	}
	if auth.Tokens.RefreshToken != "new-refresh-token" {
		t.Fatalf("stored refresh token = %q, want %q", auth.Tokens.RefreshToken, "new-refresh-token")
	}
	if auth.Tokens.IDToken != refreshedIDToken {
		t.Fatalf("stored id token = %q, want %q", auth.Tokens.IDToken, refreshedIDToken)
	}
	if auth.Tokens.AccountID != "acct_987" {
		t.Fatalf("stored account id = %q, want %q", auth.Tokens.AccountID, "acct_987")
	}
}

func TestResolveOAuthChatAuthReturnsChatGPTBaseURL(t *testing.T) {
	authHome := t.TempDir()
	authPath := filepath.Join(authHome, "auth.json")
	writeJSONFile(t, authPath, oauthAuthFile{
		Tokens: &oauthTokens{
			IDToken:      testJWT(t, map[string]any{"https://api.openai.com/auth": map[string]any{"chatgpt_account_id": "acct_321"}}),
			AccessToken:  "access-token-123",
			RefreshToken: "refresh-token-123",
			AccountID:    "acct_321",
		},
	})

	t.Setenv("OPENAI_AUTH_METHOD", "oauth")
	t.Setenv("SCIAGENT_HOME", authHome)

	_, _, baseURL, err := resolveChatAuth(io.Discard)
	if err != nil {
		t.Fatalf("resolveChatAuth() error = %v", err)
	}
	if baseURL != defaultOAuthBaseURL {
		t.Fatalf("base URL = %q, want %q", baseURL, defaultOAuthBaseURL)
	}
}

type lockedBuffer struct {
	mu  sync.Mutex
	buf strings.Builder
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

type syncValue[T any] struct {
	mu    sync.Mutex
	value T
}

func (v *syncValue[T]) Store(value T) {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.value = value
}

func (v *syncValue[T]) Load() T {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.value
}

func waitForPromptURL(t *testing.T, prompt *lockedBuffer, prefix string) string {
	t.Helper()

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		text := prompt.String()
		for _, line := range strings.Split(text, "\n") {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, prefix) {
				return line
			}
		}

		time.Sleep(10 * time.Millisecond)
	}

	t.Fatalf("timed out waiting for prompt URL with prefix %q; prompt=%q", prefix, prompt.String())
	return ""
}

func testJWT(t *testing.T, payload map[string]any) string {
	t.Helper()

	headerBytes, err := json.Marshal(map[string]string{
		"alg": "none",
		"typ": "JWT",
	})
	if err != nil {
		t.Fatalf("json.Marshal(header) error = %v", err)
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("json.Marshal(payload) error = %v", err)
	}

	return strings.Join([]string{
		base64.RawURLEncoding.EncodeToString(headerBytes),
		base64.RawURLEncoding.EncodeToString(payloadBytes),
		base64.RawURLEncoding.EncodeToString([]byte("sig")),
	}, ".")
}

func writeJSON(t *testing.T, w http.ResponseWriter, statusCode int, body map[string]any) {
	t.Helper()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	if err := json.NewEncoder(w).Encode(body); err != nil {
		t.Fatalf("json.NewEncoder().Encode() error = %v", err)
	}
}

func writeJSONFile(t *testing.T, path string, body any) {
	t.Helper()

	data, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("os.WriteFile() error = %v", err)
	}
}
