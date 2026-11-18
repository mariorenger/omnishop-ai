package main

import (
	"strings"
	"unicode/utf8"
)

// Token budgeting for small-context LLMs (e.g. 16k). Estimates are deliberately pessimistic:
// it is better to send less than to get "context length exceeded".

func estTokens(s string) int {
	if s == "" {
		return 0
	}
	ascii, other := 0, 0
	for _, r := range s {
		if r < 128 {
			ascii++
		} else {
			other++
		}
	}
	return ascii/3 + other*2/3 + 1
}

func msgTokens(m Msg) int {
	n := estTokens(m.Content) + 6
	for _, tc := range m.ToolCalls {
		n += estTokens(tc.Function.Name) + estTokens(tc.Function.Arguments) + 8
	}
	return n
}

func msgsTokens(ms []Msg) int {
	n := 0
	for _, m := range ms {
		n += msgTokens(m)
	}
	return n
}

// clipRunes cuts s to at most n bytes without splitting a UTF-8 character.
func clipRunes(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n] + "…[cắt bớt]"
}

// isResult marks synthetic "tool result" messages (text protocol) so they don't start a new turn.
func startsTurn(m Msg) bool { return m.Role == "user" && m.Meta != "result" }

func splitTurns(h []Msg) [][]Msg {
	var turns [][]Msg
	for _, m := range h {
		if startsTurn(m) || len(turns) == 0 {
			turns = append(turns, nil)
		}
		turns[len(turns)-1] = append(turns[len(turns)-1], m)
	}
	return turns
}

func clipResults(turn []Msg, limit int) []Msg {
	out := make([]Msg, len(turn))
	for i, m := range turn {
		if m.Role == "tool" || m.Meta == "result" {
			m.Content = clipRunes(m.Content, limit)
		}
		out[i] = m
	}
	return out
}

// groups after the first (user) message: assistant [+ tool/result messages...]
func dropOldestGroup(turn []Msg) []Msg {
	if len(turn) <= 2 {
		return turn
	}
	i := 1
	for i < len(turn) && turn[i].Role != "assistant" {
		i++
	}
	j := i + 1
	for j < len(turn) && turn[j].Role != "assistant" {
		j++
	}
	if i >= len(turn) || j >= len(turn) { // keep at least the newest group
		return turn
	}
	return append([]Msg{turn[0]}, turn[j:]...)
}

// fitContext returns system + as much recent history as fits in budget tokens.
func fitContext(system string, hist []Msg, budget int) []Msg {
	sys := Msg{Role: "system", Content: system}
	avail := budget - msgTokens(sys)
	if avail < 400 {
		avail = 400
	}
	turns := splitTurns(hist)
	if len(turns) == 0 {
		return []Msg{sys}
	}
	// newest turn is always kept; shrink it until it fits
	newest := turns[len(turns)-1]
	for _, lim := range []int{6000, 2500, 1000, 400, 150} {
		if msgsTokens(newest) <= avail {
			break
		}
		newest = clipResults(newest, lim)
	}
	for msgsTokens(newest) > avail {
		shorter := dropOldestGroup(newest)
		if len(shorter) == len(newest) {
			break
		}
		newest = shorter
	}
	if msgsTokens(newest) > avail { // last resort: cut the user message / final assistant text
		for i := range newest {
			if newest[i].Role == "user" || newest[i].Role == "assistant" {
				newest[i].Content = clipRunes(newest[i].Content, max(200, avail*2/len(newest)))
			}
		}
	}
	used := msgsTokens(newest)
	var kept [][]Msg
	for i := len(turns) - 2; i >= 0; i-- {
		t := clipResults(turns[i], 500) // older turns: keep the gist, not the bulk
		n := msgsTokens(t)
		if used+n > avail {
			break
		}
		used += n
		kept = append([][]Msg{t}, kept...)
	}
	out := []Msg{sys}
	for _, t := range kept {
		out = append(out, t...)
	}
	return append(out, newest...)
}

func compactHistory(h []Msg, keep int) []Msg {
	if len(h) <= keep {
		return h
	}
	cut := len(h) - keep
	for cut < len(h) && !startsTurn(h[cut]) { // never start in the middle of a turn
		cut++
	}
	if cut >= len(h) {
		return h
	}
	return h[cut:]
}

var _ = strings.TrimSpace

func sortStrings(a []string) {
	for i := 1; i < len(a); i++ {
		for j := i; j > 0 && a[j] < a[j-1]; j-- {
			a[j], a[j-1] = a[j-1], a[j]
		}
	}
}
