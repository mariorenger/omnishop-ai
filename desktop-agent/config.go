package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
)

// Config is read from config.json next to the exe; env vars override.
type Config struct {
	BaseURL     string `json:"base_url"` // OpenAI-compatible endpoint, e.g. http://llm.corp:8000/v1. Empty = offline if/else bot
	APIKey      string `json:"api_key"`  // optional
	Model       string `json:"model"`
	Port        int    `json:"port"`                // 0 = default 7878, falls back to random if busy
	AutoApprove bool   `json:"auto_approve"`        // skip confirmation for write/edit/run
	CmdTimeout  int    `json:"command_timeout_sec"` // default timeout of `run_command` (default 60, max 3600)
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
	if c.CmdTimeout <= 0 {
		c.CmdTimeout = 60
	}
	if c.Port == 0 {
		c.Port = 7878
	}
	return c
}
