package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/openai/openai-go/v3/option"
)

const (
	defaultOAuthIssuer         = "https://auth.openai.com"
	defaultOAuthClientID       = "app_EMoamEEZ73f0CkXaXp7hrann"
	defaultOAuthBaseURL        = "https://chatgpt.com/backend-api/codex"
	oauthHTTPTimeout           = 30 * time.Second
	oauthBrowserLoginTimeout   = 15 * time.Minute
	oauthCallbackShutdownGrace = 5 * time.Second
	oauthAuthorizeScopes       = "openid profile email offline_access api.connectors.read api.connectors.invoke"
	oauthCallbackBindAddr      = "127.0.0.1:1455"
	oauthCallbackURLHost       = "localhost"
	oauthTokenRefreshSkew      = 30 * time.Second
	oauthLoginOriginator       = "codex_cli_rs"
)

type oauthConfig struct {
	issuer             string
	clientID           string
	refreshURL         string
	allowedWorkspaceID string
	codexHome          string
	httpClient         *http.Client
	prompt             io.Writer
}

type oauthAuthFile struct {
	OpenAIAPIKey string       `json:"OPENAI_API_KEY,omitempty"`
	Tokens       *oauthTokens `json:"tokens,omitempty"`
	LastRefresh  *time.Time   `json:"last_refresh,omitempty"`
}

type oauthTokens struct {
	IDToken      string `json:"id_token,omitempty"`
	AccessToken  string `json:"access_token,omitempty"`
	RefreshToken string `json:"refresh_token,omitempty"`
	AccountID    string `json:"account_id,omitempty"`
}

type oauthTokenSource struct {
	mu           sync.Mutex
	cfg          oauthConfig
	authFilePath string
	auth         oauthAuthFile
}

type oauthExchangeResponse struct {
	IDToken      string `json:"id_token"`
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
}

type oauthRefreshResponse struct {
	IDToken      *string `json:"id_token"`
	AccessToken  *string `json:"access_token"`
	RefreshToken *string `json:"refresh_token"`
}

type oauthPKCECodes struct {
	Verifier  string
	Challenge string
}

type oauthCallbackResult struct {
	Code  string
	State string
	Err   error
}

func resolveChatAuth(output io.Writer) ([]option.RequestOption, string, string, error) {
	authMethod := strings.ToLower(strings.TrimSpace(os.Getenv("OPENAI_AUTH_METHOD")))
	apiKey := strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))

	switch authMethod {
	case "", "api-key":
		if apiKey != "" {
			return []option.RequestOption{option.WithAPIKey(apiKey)}, "API key", defaultBaseURL, nil
		}
		if authMethod == "api-key" {
			return nil, "", "", errors.New("OPENAI_AUTH_METHOD=api-key but OPENAI_API_KEY is not set")
		}
	case "oauth":
		return resolveOAuthChatAuth(output)
	default:
		return nil, "", "", fmt.Errorf("unsupported OPENAI_AUTH_METHOD %q", authMethod)
	}

	return resolveOAuthChatAuth(output)
}

func resolveOAuthChatAuth(output io.Writer) ([]option.RequestOption, string, string, error) {
	cfg, err := oauthConfigFromEnv(output)
	if err != nil {
		return nil, "", "", err
	}

	source, err := newOAuthTokenSource(cfg)
	if err != nil {
		return nil, "", "", err
	}

	if _, err := source.ensureRequestCredential(context.Background()); err != nil {
		return nil, "", "", err
	}

	return []option.RequestOption{
		option.WithMiddleware(source.middleware()),
	}, fmt.Sprintf("OAuth (%s)", source.authFilePath), defaultOAuthBaseURL, nil
}

func oauthConfigFromEnv(output io.Writer) (oauthConfig, error) {
	issuer := strings.TrimSpace(os.Getenv("OPENAI_OAUTH_ISSUER"))
	if issuer == "" {
		issuer = defaultOAuthIssuer
	}
	issuer = strings.TrimRight(issuer, "/")

	clientID := strings.TrimSpace(os.Getenv("OPENAI_OAUTH_CLIENT_ID"))
	if clientID == "" {
		clientID = defaultOAuthClientID
	}

	refreshURL := strings.TrimSpace(os.Getenv("OPENAI_OAUTH_REFRESH_URL"))
	if refreshURL == "" {
		refreshURL = issuer + "/oauth/token"
	}

	allowedWorkspaceID := strings.TrimSpace(os.Getenv("OPENAI_OAUTH_ALLOWED_WORKSPACE_ID"))

	codexHome := strings.TrimSpace(os.Getenv("SCIAGENT_HOME"))
	if codexHome == "" {
		var err error
		codexHome, err = defaultAuthHome()
		if err != nil {
			return oauthConfig{}, err
		}
	}

	if output == nil {
		output = io.Discard
	}

	return oauthConfig{
		issuer:             issuer,
		clientID:           clientID,
		refreshURL:         refreshURL,
		allowedWorkspaceID: allowedWorkspaceID,
		codexHome:          codexHome,
		httpClient:         &http.Client{Timeout: oauthHTTPTimeout},
		prompt:             output,
	}, nil
}

func defaultAuthHome() (string, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("determine home directory for OAuth auth file: %w", err)
	}

	return filepath.Join(homeDir, ".sciagent"), nil
}

func newOAuthTokenSource(cfg oauthConfig) (*oauthTokenSource, error) {
	if cfg.issuer == "" {
		cfg.issuer = defaultOAuthIssuer
	}
	cfg.issuer = strings.TrimRight(cfg.issuer, "/")

	if cfg.clientID == "" {
		cfg.clientID = defaultOAuthClientID
	}
	if cfg.refreshURL == "" {
		cfg.refreshURL = cfg.issuer + "/oauth/token"
	}
	if cfg.codexHome == "" {
		return nil, errors.New("OAuth auth home directory is not set")
	}
	if cfg.httpClient == nil {
		cfg.httpClient = &http.Client{Timeout: oauthHTTPTimeout}
	}
	if cfg.prompt == nil {
		cfg.prompt = io.Discard
	}

	source := &oauthTokenSource{
		cfg:          cfg,
		authFilePath: filepath.Join(cfg.codexHome, "auth.json"),
	}

	if err := source.loadAuthFile(); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}

	return source, nil
}

func (s *oauthTokenSource) middleware() option.Middleware {
	return func(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
		accessToken, accountID, err := s.ensureRequestIdentity(req.Context())
		if err != nil {
			return nil, err
		}

		req.Header.Set("Authorization", "Bearer "+accessToken)
		if accountID != "" {
			req.Header.Set("ChatGPT-Account-ID", accountID)
		} else {
			req.Header.Del("ChatGPT-Account-ID")
		}
		resp, err := next(req)
		if err != nil || resp == nil || resp.StatusCode != http.StatusUnauthorized || req.GetBody == nil {
			return resp, err
		}

		if closeErr := resp.Body.Close(); closeErr != nil {
			return nil, closeErr
		}

		if err := s.refreshAfterUnauthorized(req.Context(), accessToken); err != nil {
			return nil, fmt.Errorf("refresh OAuth token after 401: %w", err)
		}

		retryReq := req.Clone(req.Context())
		retryReq.Body, err = req.GetBody()
		if err != nil {
			return nil, err
		}

		refreshedToken, refreshedAccountID, err := s.ensureRequestIdentity(req.Context())
		if err != nil {
			return nil, err
		}
		retryReq.Header.Set("Authorization", "Bearer "+refreshedToken)
		if refreshedAccountID != "" {
			retryReq.Header.Set("ChatGPT-Account-ID", refreshedAccountID)
		} else {
			retryReq.Header.Del("ChatGPT-Account-ID")
		}

		return next(retryReq)
	}
}

func (s *oauthTokenSource) ensureRequestCredential(ctx context.Context) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.ensureSessionLocked(ctx); err != nil {
		return "", err
	}

	credential := s.requestCredentialLocked()
	if credential == "" {
		return "", errors.New("OAuth login completed without an access token")
	}

	return credential, nil
}

func (s *oauthTokenSource) ensureRequestIdentity(ctx context.Context) (string, string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.ensureSessionLocked(ctx); err != nil {
		return "", "", err
	}

	credential := s.requestCredentialLocked()
	if credential == "" {
		return "", "", errors.New("OAuth login completed without an access token")
	}

	return credential, s.accountIDLocked(), nil
}

func (s *oauthTokenSource) ensureSessionLocked(ctx context.Context) error {
	accessToken := s.accessTokenLocked()
	if accessToken != "" && !oauthTokenExpiresSoon(accessToken) {
		return nil
	}

	if s.refreshTokenLocked() != "" {
		if err := s.refreshLocked(ctx); err == nil {
			if token := s.accessTokenLocked(); token != "" {
				return nil
			}
		}
	}

	if err := s.loginLocked(ctx); err != nil {
		return err
	}

	if s.accessTokenLocked() == "" {
		return errors.New("OAuth login completed without an access token")
	}

	return nil
}

func (s *oauthTokenSource) refreshAfterUnauthorized(ctx context.Context, staleCredential string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if current := s.requestCredentialLocked(); current != "" && current != staleCredential {
		return nil
	}

	if err := s.refreshLocked(ctx); err != nil {
		return err
	}

	if current := s.requestCredentialLocked(); current == "" {
		return errors.New("OAuth refresh completed without an access token")
	}

	return nil
}

func (s *oauthTokenSource) requestCredentialLocked() string {
	return s.accessTokenLocked()
}

func (s *oauthTokenSource) accessTokenLocked() string {
	if s.auth.Tokens == nil {
		return ""
	}
	return strings.TrimSpace(s.auth.Tokens.AccessToken)
}

func (s *oauthTokenSource) refreshTokenLocked() string {
	if s.auth.Tokens == nil {
		return ""
	}
	return strings.TrimSpace(s.auth.Tokens.RefreshToken)
}

func (s *oauthTokenSource) accountIDLocked() string {
	if s.auth.Tokens == nil {
		return ""
	}
	if accountID := strings.TrimSpace(s.auth.Tokens.AccountID); accountID != "" {
		return accountID
	}
	return jwtAuthClaimString(s.auth.Tokens.IDToken, "chatgpt_account_id")
}

func (s *oauthTokenSource) loadAuthFile() error {
	data, err := os.ReadFile(s.authFilePath)
	if err != nil {
		return err
	}

	var auth oauthAuthFile
	if err := json.Unmarshal(data, &auth); err != nil {
		return fmt.Errorf("parse OAuth auth file %q: %w", s.authFilePath, err)
	}

	s.auth = auth
	return nil
}

func (s *oauthTokenSource) saveAuthFileLocked() error {
	now := time.Now().UTC()
	s.auth.LastRefresh = &now

	if s.auth.Tokens != nil {
		if accountID := jwtAuthClaimString(s.auth.Tokens.IDToken, "chatgpt_account_id"); accountID != "" {
			s.auth.Tokens.AccountID = accountID
		}
	}

	if err := os.MkdirAll(filepath.Dir(s.authFilePath), 0o700); err != nil {
		return fmt.Errorf("create OAuth auth directory: %w", err)
	}

	data, err := json.MarshalIndent(s.auth, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal OAuth auth file: %w", err)
	}

	data = append(data, '\n')
	if err := os.WriteFile(s.authFilePath, data, 0o600); err != nil {
		return fmt.Errorf("write OAuth auth file %q: %w", s.authFilePath, err)
	}

	return nil
}

func (s *oauthTokenSource) loginLocked(ctx context.Context) error {
	loginCtx, cancel := context.WithTimeout(ctx, oauthBrowserLoginTimeout)
	defer cancel()

	pkce, err := newOAuthPKCECodes()
	if err != nil {
		return err
	}

	state, err := randomBase64URL(32)
	if err != nil {
		return fmt.Errorf("generate OAuth state: %w", err)
	}

	callbackResultCh := make(chan oauthCallbackResult, 1)
	callbackErrCh := make(chan error, 1)
	listener, server, redirectURI, err := newOAuthCallbackServer(state, callbackResultCh)
	if err != nil {
		return err
	}
	defer func() {
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), oauthCallbackShutdownGrace)
		defer shutdownCancel()
		_ = server.Shutdown(shutdownCtx)
	}()

	go func() {
		if err := server.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
			select {
			case callbackErrCh <- err:
			default:
			}
		}
	}()

	authURL := buildOAuthAuthorizeURL(s.cfg.issuer, s.cfg.clientID, redirectURI, pkce, state, s.cfg.allowedWorkspaceID)
	printBrowserLoginPrompt(s.cfg.prompt, authURL)

	var callback oauthCallbackResult
	select {
	case callback = <-callbackResultCh:
	case err := <-callbackErrCh:
		return fmt.Errorf("run OAuth callback server: %w", err)
	case <-loginCtx.Done():
		return fmt.Errorf("OAuth browser login did not complete: %w", loginCtx.Err())
	}

	if callback.Err != nil {
		return callback.Err
	}
	if callback.State != state {
		return errors.New("OAuth callback state did not match the login request")
	}

	exchanged, err := s.exchangeAuthorizationCode(loginCtx, callback.Code, pkce.Verifier, redirectURI)
	if err != nil {
		return err
	}

	s.auth.Tokens = &oauthTokens{
		IDToken:      exchanged.IDToken,
		AccessToken:  exchanged.AccessToken,
		RefreshToken: exchanged.RefreshToken,
		AccountID:    jwtAuthClaimString(exchanged.IDToken, "chatgpt_account_id"),
	}
	s.auth.OpenAIAPIKey = ""

	return s.saveAuthFileLocked()
}

func (s *oauthTokenSource) refreshLocked(ctx context.Context) error {
	refreshToken := s.refreshTokenLocked()
	if refreshToken == "" {
		return errors.New("OAuth session does not include a refresh token")
	}

	body, err := json.Marshal(map[string]string{
		"client_id":     s.cfg.clientID,
		"grant_type":    "refresh_token",
		"refresh_token": refreshToken,
	})
	if err != nil {
		return fmt.Errorf("marshal OAuth refresh request: %w", err)
	}

	resp, data, err := s.doRequest(ctx, http.MethodPost, s.cfg.refreshURL, "application/json", body)
	if err != nil {
		return err
	}
	if !isHTTPSuccess(resp.StatusCode) {
		return fmt.Errorf("OAuth token refresh failed with status %s: %s", resp.Status, oauthErrorMessage(data))
	}

	var refreshed oauthRefreshResponse
	if err := json.Unmarshal(data, &refreshed); err != nil {
		return fmt.Errorf("parse OAuth refresh response: %w", err)
	}

	if s.auth.Tokens == nil {
		s.auth.Tokens = &oauthTokens{}
	}
	if refreshed.IDToken != nil {
		s.auth.Tokens.IDToken = *refreshed.IDToken
	}
	if refreshed.AccessToken != nil {
		s.auth.Tokens.AccessToken = *refreshed.AccessToken
	}
	if refreshed.RefreshToken != nil {
		s.auth.Tokens.RefreshToken = *refreshed.RefreshToken
	}

	if s.auth.Tokens.AccessToken == "" {
		return errors.New("OAuth refresh response did not include an access token")
	}

	return s.saveAuthFileLocked()
}

func (s *oauthTokenSource) exchangeAuthorizationCode(ctx context.Context, authorizationCode string, codeVerifier string, redirectURI string) (oauthExchangeResponse, error) {
	form := url.Values{
		"grant_type":    []string{"authorization_code"},
		"code":          []string{authorizationCode},
		"redirect_uri":  []string{redirectURI},
		"client_id":     []string{s.cfg.clientID},
		"code_verifier": []string{codeVerifier},
	}

	url := s.cfg.issuer + "/oauth/token"
	resp, data, err := s.doRequest(ctx, http.MethodPost, url, "application/x-www-form-urlencoded", []byte(form.Encode()))
	if err != nil {
		return oauthExchangeResponse{}, err
	}
	if !isHTTPSuccess(resp.StatusCode) {
		return oauthExchangeResponse{}, fmt.Errorf("OAuth token exchange failed with status %s: %s", resp.Status, oauthErrorMessage(data))
	}

	var exchanged oauthExchangeResponse
	if err := json.Unmarshal(data, &exchanged); err != nil {
		return oauthExchangeResponse{}, fmt.Errorf("parse OAuth token exchange response: %w", err)
	}
	if exchanged.AccessToken == "" {
		return oauthExchangeResponse{}, errors.New("OAuth token exchange response did not include an access token")
	}

	return exchanged, nil
}

func (s *oauthTokenSource) doRequest(ctx context.Context, method string, url string, contentType string, body []byte) (*http.Response, []byte, error) {
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
	if err != nil {
		return nil, nil, err
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}

	resp, err := s.cfg.httpClient.Do(req)
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, err
	}

	return resp, data, nil
}

func newOAuthCallbackServer(expectedState string, callbackResultCh chan<- oauthCallbackResult) (net.Listener, *http.Server, string, error) {
	listener, err := net.Listen("tcp", oauthCallbackBindAddr)
	if err != nil {
		return nil, nil, "", fmt.Errorf("start OAuth callback listener: %w", err)
	}

	tcpAddr, ok := listener.Addr().(*net.TCPAddr)
	if !ok {
		_ = listener.Close()
		return nil, nil, "", errors.New("determine OAuth callback listener port")
	}

	redirectURI := fmt.Sprintf("http://%s:%d/auth/callback", oauthCallbackURLHost, tcpAddr.Port)
	mux := http.NewServeMux()
	mux.HandleFunc("/auth/callback", func(w http.ResponseWriter, r *http.Request) {
		result := oauthCallbackResult{
			Code:  r.URL.Query().Get("code"),
			State: r.URL.Query().Get("state"),
		}

		statusCode := http.StatusOK
		responseBody := "Authentication complete. You may close this window."

		if authErr := r.URL.Query().Get("error"); authErr != "" {
			statusCode = http.StatusBadRequest
			description := r.URL.Query().Get("error_description")
			if description != "" {
				result.Err = fmt.Errorf("OAuth authorization failed: %s: %s", authErr, description)
			} else {
				result.Err = fmt.Errorf("OAuth authorization failed: %s", authErr)
			}
			responseBody = result.Err.Error()
		} else if result.Code == "" {
			statusCode = http.StatusBadRequest
			result.Err = errors.New("OAuth callback did not include an authorization code")
			responseBody = result.Err.Error()
		} else if result.State == "" {
			statusCode = http.StatusBadRequest
			result.Err = errors.New("OAuth callback did not include a state value")
			responseBody = result.Err.Error()
		} else if result.State != expectedState {
			statusCode = http.StatusBadRequest
			result.Err = errors.New("OAuth callback state did not match the login request")
			responseBody = result.Err.Error()
		}

		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(statusCode)
		_, _ = io.WriteString(w, responseBody)

		select {
		case callbackResultCh <- result:
		default:
		}
	})

	return listener, &http.Server{Handler: mux}, redirectURI, nil
}

func newOAuthPKCECodes() (oauthPKCECodes, error) {
	verifier, err := randomBase64URL(32)
	if err != nil {
		return oauthPKCECodes{}, fmt.Errorf("generate OAuth PKCE verifier: %w", err)
	}

	sum := sha256.Sum256([]byte(verifier))
	return oauthPKCECodes{
		Verifier:  verifier,
		Challenge: base64.RawURLEncoding.EncodeToString(sum[:]),
	}, nil
}

func randomBase64URL(size int) (string, error) {
	buf := make([]byte, size)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}

	return base64.RawURLEncoding.EncodeToString(buf), nil
}

func buildOAuthAuthorizeURL(issuer string, clientID string, redirectURI string, pkce oauthPKCECodes, state string, allowedWorkspaceID string) string {
	query := []struct {
		key   string
		value string
	}{
		{key: "response_type", value: "code"},
		{key: "client_id", value: clientID},
		{key: "redirect_uri", value: redirectURI},
		{key: "scope", value: oauthAuthorizeScopes},
		{key: "code_challenge", value: pkce.Challenge},
		{key: "code_challenge_method", value: "S256"},
		{key: "id_token_add_organizations", value: "true"},
		{key: "codex_cli_simplified_flow", value: "true"},
		{key: "state", value: state},
		{key: "originator", value: oauthLoginOriginator},
	}
	if allowedWorkspaceID != "" {
		query = append(query, struct {
			key   string
			value string
		}{key: "allowed_workspace_id", value: allowedWorkspaceID})
	}

	parts := make([]string, 0, len(query))
	for _, entry := range query {
		parts = append(parts, entry.key+"="+oauthAuthorizeValue(entry.value))
	}

	return issuer + "/oauth/authorize?" + strings.Join(parts, "&")
}

func oauthTokenExpiresSoon(token string) bool {
	expiresAt, ok := jwtExpiration(token)
	if !ok {
		return false
	}

	return time.Until(expiresAt) <= oauthTokenRefreshSkew
}

func jwtExpiration(token string) (time.Time, bool) {
	claims, ok := jwtClaims(token)
	if !ok {
		return time.Time{}, false
	}

	rawExp, ok := claims["exp"]
	if !ok {
		return time.Time{}, false
	}

	switch value := rawExp.(type) {
	case float64:
		return time.Unix(int64(value), 0), true
	case int64:
		return time.Unix(value, 0), true
	case json.Number:
		seconds, err := value.Int64()
		if err != nil {
			return time.Time{}, false
		}
		return time.Unix(seconds, 0), true
	default:
		return time.Time{}, false
	}
}

func jwtAuthClaimString(token string, key string) string {
	claims, ok := jwtClaims(token)
	if !ok {
		return ""
	}

	if authClaims, ok := claims["https://api.openai.com/auth"].(map[string]any); ok {
		if value, ok := authClaims[key].(string); ok {
			return value
		}
	}
	if value, ok := claims[key].(string); ok {
		return value
	}

	return ""
}

func jwtClaims(token string) (map[string]any, bool) {
	parts := strings.Split(token, ".")
	if len(parts) < 2 {
		return nil, false
	}

	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, false
	}

	var claims map[string]any
	if err := json.Unmarshal(payload, &claims); err != nil {
		return nil, false
	}

	return claims, true
}

func printBrowserLoginPrompt(output io.Writer, authURL string) {
	if output == nil {
		return
	}

	fmt.Fprintf(output,
		"\nOpen this URL in your browser to sign in:\n%s\n\nWaiting for the browser callback to complete...\n\n",
		authURL,
	)
}

func oauthErrorMessage(data []byte) string {
	trimmed := strings.TrimSpace(string(data))
	if trimmed == "" {
		return "empty response body"
	}

	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		return trimmed
	}

	errorDescription := jsonString(payload["error_description"])
	message := jsonString(payload["message"])
	code := jsonString(payload["code"])

	switch errorValue := payload["error"].(type) {
	case string:
		if errorDescription != "" {
			return errorValue + ": " + errorDescription
		}
		return errorValue
	case map[string]any:
		if code == "" {
			code = jsonString(errorValue["code"])
		}
		if message == "" {
			message = jsonString(errorValue["message"])
		}
	}

	if message != "" && code != "" {
		return code + ": " + message
	}
	if message != "" {
		return message
	}
	if errorDescription != "" {
		return errorDescription
	}
	if code != "" {
		return code
	}

	return trimmed
}

func jsonString(value any) string {
	text, _ := value.(string)
	return text
}

func isHTTPSuccess(statusCode int) bool {
	return statusCode >= 200 && statusCode < 300
}

func oauthAuthorizeValue(value string) string {
	return strings.ReplaceAll(url.QueryEscape(value), "+", "%20")
}
