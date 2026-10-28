// claude-helper: loopback HTTP wrapper around `claude -p` for the RAG sandbox.
package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	defaultPort = "8765"
	maxBody     = 4 << 20
	claudeLimit = 300 * time.Second
)

type server struct {
	token string
	mu    sync.Mutex // one claude call at a time
}

func (s *server) authorized(r *http.Request) bool {
	if s.token == "" {
		return false // an empty token must never match an empty bearer
	}
	got := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	if got == r.Header.Get("Authorization") {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(got), []byte(s.token)) == 1
}

func (s *server) handleComplete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !s.authorized(r) {
		log.Printf("rejected unauthorized request from %s", r.RemoteAddr)
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}
	var req struct {
		Prompt string `json:"prompt"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBody)).Decode(&req); err != nil || req.Prompt == "" {
		http.Error(w, "bad request: need JSON {\"prompt\": \"...\"}", http.StatusBadRequest)
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	start := time.Now()
	text, err := runClaude(r.Context(), req.Prompt)
	if err != nil {
		log.Printf("claude failed after %s: %v", time.Since(start).Round(time.Millisecond), err)
		http.Error(w, "claude failed", http.StatusBadGateway)
		return
	}
	log.Printf("claude ok in %s (prompt %d bytes, reply %d bytes)", time.Since(start).Round(time.Millisecond), len(req.Prompt), len(text))
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"text": text})
}

// claudeArgs locks the child down because the prompt embeds untrusted document text: no tools,
// no MCP servers, no user/project settings or hooks. Each flag is its own argv item.
func claudeArgs(model string) []string {
	args := []string{"-p", "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence"}
	if model != "" {
		args = append(args, "--model", model)
	}
	return args
}

// runClaude passes the prompt on stdin only.
func runClaude(parent context.Context, prompt string) (string, error) {
	ctx, cancel := context.WithTimeout(parent, claudeLimit)
	defer cancel()

	cmd := exec.CommandContext(ctx, "claude", claudeArgs(os.Getenv("CLAUDE_MODEL"))...)
	cmd.Stdin = strings.NewReader(prompt)
	cmd.Env = append(withoutAPIKey(os.Environ()), "CLAUDE_CODE_DISABLE_CLAUDE_MDS=1") // keep CLAUDE.md files out of the prompt
	dir, err := os.MkdirTemp("", "claude-helper-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(dir)
	cmd.Dir = dir
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Run(); err != nil {
		return "", errors.New(err.Error() + ": " + truncate(errb.String(), 300))
	}
	return strings.TrimSpace(out.String()), nil
}

func withoutAPIKey(env []string) []string {
	kept := make([]string, 0, len(env))
	for _, kv := range env {
		if !strings.HasPrefix(kv, "ANTHROPIC_API_KEY=") {
			kept = append(kept, kv)
		}
	}
	return kept
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// loadToken reads the token file, creating it (mode 0600) with a random token if absent.
func loadToken(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err == nil {
		tok := strings.TrimSpace(string(b))
		if tok == "" {
			return "", errors.New("token file is empty: " + path)
		}
		return tok, nil
	}
	if !os.IsNotExist(err) {
		return "", err
	}
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", err
	}
	tok := hex.EncodeToString(raw)
	if err := os.WriteFile(path, []byte(tok+"\n"), 0o600); err != nil {
		return "", err
	}
	return tok, nil
}

func newMux(s *server) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/complete", s.handleComplete)
	return mux
}

func main() {
	exe, err := os.Executable()
	if err != nil {
		log.Fatal(err)
	}
	tok, err := loadToken(filepath.Join(filepath.Dir(exe), ".token"))
	if err != nil {
		log.Fatal(err)
	}
	port := defaultPort
	if p := os.Getenv("HELPER_PORT"); p != "" {
		port = p
	}
	addr := "127.0.0.1:" + port // loopback only, never configurable
	srv := &http.Server{
		Addr:              addr,
		Handler:           newMux(&server{token: tok}),
		ReadHeaderTimeout: 10 * time.Second,
		WriteTimeout:      claudeLimit + 30*time.Second,
	}
	log.Printf("listening on %s", addr)
	log.Fatal(srv.ListenAndServe())
}
