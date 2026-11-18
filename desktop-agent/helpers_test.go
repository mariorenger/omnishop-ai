package main

import (
	"encoding/json"
	"io"
	"os"
	"strconv"
	"testing"
	"time"
)

func mustWrite(t *testing.T, p, s string) {
	t.Helper()
	must(t, os.WriteFile(p, []byte(s), 0o644))
}
func copyFile(t *testing.T, from, to string) {
	t.Helper()
	in, err := os.Open(from)
	must(t, err)
	defer in.Close()
	out, err := os.Create(to)
	must(t, err)
	defer out.Close()
	_, err = io.Copy(out, in)
	must(t, err)
}
func itoa(i int) string { return strconv.Itoa(i) }
func atoi(s string) int { n, _ := strconv.Atoi(s); return n }

// denyAll answers every approval request with "deny" until stop() is called.
func denyAll(s *Server) (stop func()) {
	done := make(chan struct{})
	go func() {
		for {
			select {
			case <-done:
				return
			default:
			}
			s.mu.Lock()
			for id, ch := range s.pending {
				select {
				case ch <- false:
				default:
				}
				delete(s.pending, id)
			}
			s.mu.Unlock()
			time.Sleep(2 * time.Millisecond)
		}
	}()
	return func() { close(done) }
}

func jsonMarshalImpl(v any) ([]byte, error) { return json.Marshal(v) }
