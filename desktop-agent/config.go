package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
)

// Config is read from config.json next to the exe; env vars override.
type Config struct {
	BaseURL         string `json:"base_url"` // OpenAI-compatible endpoint, e.g. http://llm.corp:8000/v1. Empty = offline if/else bot
	APIKey          string `json:"api_key"`  // optional
	Model           string `json:"model"`
	Port            int    `json:"port"`              // 0 = default 7878, falls back to random if busy
	AutoApprove     bool   `json:"auto_approve"`      // skip confirmation for write/edit/run
	ContextTokens   int    `json:"context_tokens"`    // LLM context window (default 16000)
	MaxOutputTokens int    `json:"max_output_tokens"` // reserved for the answer (default 1500)
	ToolMode        string `json:"tool_mode"`         // "text" (default, works with any model) | "native" | "auto"
	SearchProvider  string `json:"search_provider"`   // duckduckgo (default) | google | searxng
	GoogleAPIKey    string `json:"google_api_key"`
	GoogleCX        string `json:"google_cx"`
	SearxngURL      string `json:"searxng_url"`
	AccessToken     string `json:"access_token"`        // optional fixed URL token (>=16 chars) for running as a service; empty = random each start
	AlertWebhook    string `json:"alert_webhook"`       // POST {"text":...} on task alerts (Teams/Slack/Zalo bridge)
	CmdTimeout      int    `json:"command_timeout_sec"` // default timeout of `run_command` (default 60, max 3600)
}

func loadConfig() Config {
	c := Config{BaseURL: "", Model: "", Port: 7878, CmdTimeout: 60}
	if exe, err := os.Executable(); err == nil {
		if b, err := os.ReadFile(filepath.Join(filepath.Dir(exe), "config.json")); err == nil {
			_ = json.Unmarshal(b, &c)
		}
	}
	if b, err := os.ReadFile("config.json"); err == nil {
		_ = json.Unmarshal(b, &c)
	}
	if v := os.Getenv("LLM_BASE_URL"); v != "" {
		c.BaseURL = v
	}
	if v := os.Getenv("LLM_API_KEY"); v != "" {
		c.APIKey = v
	}
	if v := os.Getenv("LLM_MODEL"); v != "" {
		c.Model = v
	}
	if v := os.Getenv("AGENT_PORT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			c.Port = n
		}
	}
	if c.ContextTokens <= 0 {
		c.ContextTokens = 16000
	}
	if c.MaxOutputTokens <= 0 {
		c.MaxOutputTokens = 1500
	}
	if c.CmdTimeout <= 0 {
		c.CmdTimeout = 60
	}
	if c.Port == 0 {
		c.Port = 7878
	}
	return c
}
