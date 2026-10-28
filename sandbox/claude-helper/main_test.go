package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fakeClaude puts a stand-in `claude` first on PATH; it records argv and env, then echoes stdin.
func fakeClaude(t *testing.T) (argsFile string) {
	t.Helper()
	dir := t.TempDir()
	argsFile = filepath.Join(dir, "args.txt")
	script := "#!/bin/sh\n" +
		"printf 'arg:%s\\n' \"$@\" > \"" + argsFile + "\"\n" +
		"echo \"key:${ANTHROPIC_API_KEY-unset}\" >> \"" + argsFile + "\"\n" +
		"echo \"mds:${CLAUDE_CODE_DISABLE_CLAUDE_MDS-unset}\" >> \"" + argsFile + "\"\n" +
		"cat\n"
	if err := os.WriteFile(filepath.Join(dir, "claude"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	return argsFile
}

// lockdown is the argv (one item per line) that disables tools, MCP servers and settings sources.
const lockdown = "arg:-p\narg:--tools\narg:\narg:--strict-mcp-config\narg:--setting-sources\narg:\narg:--no-session-persistence\n"

func TestComplete(t *testing.T) {
	const token = "s3cret"
	tests := []struct {
		name       string
		method     string
		auth       string
		body       string
		wantStatus int
		wantText   string
	}{
		{"no auth header", http.MethodPost, "", `{"prompt":"hi"}`, http.StatusUnauthorized, ""},
		{"wrong token", http.MethodPost, "Bearer nope", `{"prompt":"hi"}`, http.StatusUnauthorized, ""},
		{"not bearer scheme", http.MethodPost, token, `{"prompt":"hi"}`, http.StatusUnauthorized, ""},
		{"GET rejected", http.MethodGet, "Bearer " + token, "", http.StatusMethodNotAllowed, ""},
		{"empty prompt", http.MethodPost, "Bearer " + token, `{"prompt":""}`, http.StatusBadRequest, ""},
		{"bad json", http.MethodPost, "Bearer " + token, `nope`, http.StatusBadRequest, ""},
		{"prompt over stdin", http.MethodPost, "Bearer " + token, `{"prompt":"line1\nline2 $(rm -rf /)"}`, http.StatusOK, "line1\nline2 $(rm -rf /)"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			argsFile := fakeClaude(t)
			t.Setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
			t.Setenv("CLAUDE_MODEL", "")
			ts := httptest.NewServer(newMux(&server{token: token}))
			defer ts.Close()

			req, _ := http.NewRequest(tc.method, ts.URL+"/complete", strings.NewReader(tc.body))
			if tc.auth != "" {
				req.Header.Set("Authorization", tc.auth)
			}
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != tc.wantStatus {
				t.Fatalf("status = %d, want %d", resp.StatusCode, tc.wantStatus)
			}
			rec, readErr := os.ReadFile(argsFile)
			if tc.wantStatus != http.StatusOK {
				if readErr == nil {
					t.Fatalf("claude must not run on a rejected request, got %q", rec)
				}
				return
			}
			var got struct{ Text string }
			if err := decode(resp, &got); err != nil {
				t.Fatal(err)
			}
			if got.Text != tc.wantText {
				t.Errorf("text = %q, want %q", got.Text, tc.wantText)
			}
			if strings.Contains(string(rec), "line1") {
				t.Errorf("prompt leaked into argv: %q", rec)
			}
			if !strings.Contains(string(rec), "key:unset") {
				t.Errorf("ANTHROPIC_API_KEY not removed: %q", rec)
			}
			if !strings.Contains(string(rec), "mds:1") {
				t.Errorf("CLAUDE.md loading not disabled: %q", rec)
			}
			if strings.Contains(string(rec), "--bare") || strings.Contains(string(rec), "--model") {
				t.Errorf("unexpected flags: %q", rec)
			}
			if !strings.Contains(string(rec), lockdown) {
				t.Errorf("lockdown flags missing: %q", rec)
			}
		})
	}
}

func TestAuthorized(t *testing.T) {
	tests := []struct {
		name   string
		token  string
		header string
		want   bool
	}{
		{"match", "s3cret", "Bearer s3cret", true},
		{"mismatch", "s3cret", "Bearer nope", false},
		{"empty server token, empty bearer", "", "Bearer ", false},
		{"empty server token, no header", "", "", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodPost, "/complete", nil)
			if tc.header != "" {
				r.Header.Set("Authorization", tc.header)
			}
			if got := (&server{token: tc.token}).authorized(r); got != tc.want {
				t.Errorf("authorized = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestModelFlag(t *testing.T) {
	argsFile := fakeClaude(t)
	t.Setenv("CLAUDE_MODEL", "sonnet")
	if _, err := runClaude(t.Context(), "x"); err != nil {
		t.Fatal(err)
	}
	rec, _ := os.ReadFile(argsFile)
	if !strings.Contains(string(rec), "arg:--model\narg:sonnet\n") {
		t.Errorf("args = %q", rec)
	}
}

func TestLoadToken(t *testing.T) {
	tests := []struct {
		name    string
		content *string
		dir     bool // a directory at the path makes ReadFile fail with an error other than not-exist
		wantErr bool
	}{
		{"existing token", ptr("abc123\n"), false, false},
		{"empty file rejected", ptr(""), false, true},
		{"whitespace file rejected", ptr(" \n"), false, true},
		{"missing file generated", nil, false, false},
		{"non-NotExist read error fails", nil, true, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), ".token")
			if tc.dir {
				if err := os.Mkdir(path, 0o700); err != nil {
					t.Fatal(err)
				}
			}
			if tc.content != nil {
				if err := os.WriteFile(path, []byte(*tc.content), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			tok, err := loadToken(path)
			if (err != nil) != tc.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tc.wantErr)
			}
			if !tc.wantErr && tok == "" {
				t.Error("empty token returned")
			}
		})
	}
}

func ptr(s string) *string { return &s }

func decode(resp *http.Response, v any) error { return json.NewDecoder(resp.Body).Decode(v) }
