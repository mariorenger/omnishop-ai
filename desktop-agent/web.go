package main

import (
	"context"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
)

// Web tools: web_search (DuckDuckGo HTML by default, or Google Programmable Search, or a
// SearXNG instance) and fetch_url (page text). Needs internet / proxy access from this PC.

type webCfg struct {
	Provider  string // duckduckgo | google | searxng
	GoogleKey string
	GoogleCX  string
	SearxURL  string
}

var (
	webConf    webCfg
	ddgURL     = "https://html.duckduckgo.com/html/"
	googleURL  = "https://www.googleapis.com/customsearch/v1"
	webClient  = &http.Client{Timeout: 20 * time.Second}
	maxPageLen = 2 << 20
)

type searchHit struct{ Title, URL, Snippet string }

func httpGet(ctx context.Context, u string, hdr map[string]string) ([]byte, string, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DesktopAgent/1.0")
	req.Header.Set("Accept-Language", "vi,en;q=0.8")
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := webClient.Do(req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, int64(maxPageLen)))
	if resp.StatusCode >= 400 {
		return b, resp.Header.Get("Content-Type"), fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	return b, resp.Header.Get("Content-Type"), nil
}

var (
	reDDGLink = regexp.MustCompile(`(?s)<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>`)
	reDDGSnip = regexp.MustCompile(`(?s)<a[^>]+class="result__snippet"[^>]*>(.*?)</a>`)
	reTag     = regexp.MustCompile(`(?s)<[^>]+>`)
	reScript  = regexp.MustCompile(`(?is)<(script|style|noscript|svg|head|nav|footer|form)[^>]*>.*?</(script|style|noscript|svg|head|nav|footer|form)>`)
	reBlockEl = regexp.MustCompile(`(?i)</?(p|div|br|li|tr|h[1-6]|article|section|ul|ol|table)[^>]*>`)
	reSpaces  = regexp.MustCompile(`[ \t\r\f\v]+`)
	reNL      = regexp.MustCompile(`\n\s*\n+`)
	reTitle   = regexp.MustCompile(`(?is)<title[^>]*>(.*?)</title>`)
)

func cleanHTML(s string) string {
	return strings.TrimSpace(html.UnescapeString(reTag.ReplaceAllString(s, "")))
}

func ddgRealURL(href string) string {
	href = html.UnescapeString(href)
	if strings.HasPrefix(href, "//") {
		href = "https:" + href
	}
	if u, err := url.Parse(href); err == nil && strings.Contains(u.Host, "duckduckgo.com") && u.Path == "/l/" {
		if t := u.Query().Get("uddg"); t != "" {
			return t
		}
	}
	return href
}

func searchDDG(ctx context.Context, q string) ([]searchHit, error) {
	b, _, err := httpGet(ctx, ddgURL+"?q="+url.QueryEscape(q)+"&kl=vn-vi", nil)
	if err != nil {
		return nil, err
	}
	links := reDDGLink.FindAllStringSubmatch(string(b), -1)
	snips := reDDGSnip.FindAllStringSubmatch(string(b), -1)
	var out []searchHit
	for i, l := range links {
		h := searchHit{Title: cleanHTML(l[2]), URL: ddgRealURL(l[1])}
		if i < len(snips) {
			h.Snippet = cleanHTML(snips[i][1])
		}
		out = append(out, h)
	}
	return out, nil
}

func searchGoogle(ctx context.Context, q string) ([]searchHit, error) {
	if webConf.GoogleKey == "" || webConf.GoogleCX == "" {
		return nil, fmt.Errorf("thiếu google_api_key / google_cx trong config.json")
	}
	u := fmt.Sprintf("%s?key=%s&cx=%s&num=8&q=%s", googleURL, url.QueryEscape(webConf.GoogleKey), url.QueryEscape(webConf.GoogleCX), url.QueryEscape(q))
	b, _, err := httpGet(ctx, u, nil)
	if err != nil {
		return nil, fmt.Errorf("%v: %s", err, clip(string(b), 200))
	}
	var r struct {
		Items []struct{ Title, Link, Snippet string } `json:"items"`
	}
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, err
	}
	var out []searchHit
	for _, it := range r.Items {
		out = append(out, searchHit{it.Title, it.Link, it.Snippet})
	}
	return out, nil
}

func searchSearx(ctx context.Context, q string) ([]searchHit, error) {
	if webConf.SearxURL == "" {
		return nil, fmt.Errorf("thiếu searxng_url trong config.json")
	}
	b, _, err := httpGet(ctx, strings.TrimRight(webConf.SearxURL, "/")+"/search?format=json&q="+url.QueryEscape(q), nil)
	if err != nil {
		return nil, err
	}
	var r struct {
		Results []struct{ Title, URL, Content string } `json:"results"`
	}
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, err
	}
	var out []searchHit
	for _, it := range r.Results {
		out = append(out, searchHit{it.Title, it.URL, it.Content})
	}
	return out, nil
}

func webSearch(ctx context.Context, q string) string {
	q = strings.TrimSpace(q)
	if q == "" {
		return "error: thiếu từ khóa tìm kiếm"
	}
	var hits []searchHit
	var err error
	switch strings.ToLower(webConf.Provider) {
	case "google":
		hits, err = searchGoogle(ctx, q)
	case "searxng":
		hits, err = searchSearx(ctx, q)
	default:
		hits, err = searchDDG(ctx, q)
	}
	if err != nil {
		return "error: tìm web thất bại: " + err.Error() + " (máy này có ra được internet/proxy không?)"
	}
	if len(hits) == 0 {
		return "no matches"
	}
	if len(hits) > 8 {
		hits = hits[:8]
	}
	var sb strings.Builder
	for i, h := range hits {
		fmt.Fprintf(&sb, "%d. %s\n   %s\n   %s\n", i+1, h.Title, h.URL, clip(h.Snippet, 300))
	}
	return sb.String()
}

// pageText turns an HTML page into readable text.
func pageText(raw string) (title, text string) {
	if m := reTitle.FindStringSubmatch(raw); m != nil {
		title = cleanHTML(m[1])
	}
	s := reScript.ReplaceAllString(raw, " ")
	s = reBlockEl.ReplaceAllString(s, "\n")
	s = reTag.ReplaceAllString(s, " ")
	s = html.UnescapeString(s)
	s = reSpaces.ReplaceAllString(s, " ")
	lines := strings.Split(s, "\n")
	for i := range lines {
		lines[i] = strings.TrimSpace(lines[i])
	}
	s = reNL.ReplaceAllString(strings.Join(lines, "\n"), "\n\n")
	return title, strings.TrimSpace(s)
}

func fetchURL(ctx context.Context, rawURL string, offset int) string {
	u, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return "error: URL không hợp lệ (cần http:// hoặc https://)"
	}
	b, ct, err := httpGet(ctx, u.String(), nil)
	if err != nil {
		return "error: không tải được trang: " + err.Error()
	}
	body := string(b)
	if strings.Contains(strings.ToLower(ct), "html") || strings.Contains(body[:min(len(body), 400)], "<html") {
		title, text := pageText(body)
		if title != "" {
			text = "# " + title + "\n\n" + text
		}
		body = text
	}
	if strings.TrimSpace(body) == "" {
		return "(trang không có nội dung chữ)"
	}
	return window(body, offset, rawURL)
}
