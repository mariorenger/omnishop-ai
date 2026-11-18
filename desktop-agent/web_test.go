package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

const ddgPage = `<html><body>
<div class="result"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fvnexpress.net%2Fgia-vang&amp;rut=abc">Giá vàng hôm nay &amp; tỷ giá</a>
<a class="result__snippet" href="x">Vàng SJC tăng <b>mạnh</b> phiên sáng nay</a></div>
<div class="result"><a rel="nofollow" class="result__a" href="https://tuoitre.vn/tin">Tuổi Trẻ: tin mới</a>
<a class="result__snippet" href="y">Cập nhật liên tục</a></div></body></html>`

const articlePage = `<html><head><title>Giá vàng &amp; USD</title><style>.x{color:red}</style><script>var secret=1;</script></head>
<body><nav>Menu Trang chủ</nav><article><h1>Vàng tăng mạnh</h1><p>Giá vàng SJC lên <b>85 triệu</b> đồng/lượng.</p>
<ul><li>Mua vào: 84,5</li><li>Bán ra: 85</li></ul></article><footer>Bản quyền</footer></body></html>`

func webMock(t *testing.T) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/html/":
			if r.URL.Query().Get("q") == "" {
				http.Error(w, "no q", 400)
				return
			}
			w.Write([]byte(ddgPage))
		case "/google":
			if r.URL.Query().Get("key") != "KEY" || r.URL.Query().Get("cx") != "CX" {
				http.Error(w, `{"error":"bad key"}`, 403)
				return
			}
			w.Write([]byte(`{"items":[{"title":"Kết quả G","link":"https://g.example/1","snippet":"mô tả G"}]}`))
		case "/search":
			w.Write([]byte(`{"results":[{"title":"Searx hit","url":"https://s.example","content":"nội dung S"}]}`))
		case "/article":
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Write([]byte(articlePage))
		case "/plain":
			w.Header().Set("Content-Type", "text/plain")
			w.Write([]byte("văn bản thuần"))
		case "/empty":
			w.Header().Set("Content-Type", "text/html")
			w.Write([]byte("<html><body><script>x</script></body></html>"))
		default:
			http.Error(w, "boom", 500)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestWebSearchProviders(t *testing.T) {
	srv := webMock(t)
	oldD, oldG, oldW := ddgURL, googleURL, webConf
	defer func() { ddgURL, googleURL, webConf = oldD, oldG, oldW }()
	ddgURL, googleURL = srv.URL+"/html/", srv.URL+"/google"

	webConf = webCfg{}
	out := webSearch(t.Context(), "giá vàng")
	wantHas(t, out, "1. Giá vàng hôm nay & tỷ giá")
	wantHas(t, out, "https://vnexpress.net/gia-vang") // DDG redirect unwrapped
	wantHas(t, out, "Vàng SJC tăng mạnh phiên sáng nay")
	wantHas(t, out, "2. Tuổi Trẻ")
	if strings.Contains(out, "duckduckgo.com/l/") || strings.Contains(out, "<b>") {
		t.Errorf("redirect/HTML leaked: %s", out)
	}

	webConf = webCfg{Provider: "google", GoogleKey: "KEY", GoogleCX: "CX"}
	wantHas(t, webSearch(t.Context(), "x"), "https://g.example/1")
	webConf = webCfg{Provider: "google", GoogleKey: "BAD", GoogleCX: "CX"}
	wantHas(t, webSearch(t.Context(), "x"), "thất bại")
	webConf = webCfg{Provider: "google"}
	wantHas(t, webSearch(t.Context(), "x"), "google_api_key")

	webConf = webCfg{Provider: "searxng", SearxURL: srv.URL}
	wantHas(t, webSearch(t.Context(), "x"), "Searx hit")
	webConf = webCfg{Provider: "searxng"}
	wantHas(t, webSearch(t.Context(), "x"), "searxng_url")

	wantHas(t, webSearch(t.Context(), "  "), "thiếu từ khóa")
	ddgURL = "http://127.0.0.1:1/" // unreachable → readable error, no panic/hang
	webConf = webCfg{}
	wantHas(t, webSearch(t.Context(), "x"), "internet")
}

func TestFetchURL(t *testing.T) {
	srv := webMock(t)
	out := fetchURL(t.Context(), srv.URL+"/article", 0)
	for _, w := range []string{"# Giá vàng & USD", "Vàng tăng mạnh", "85 triệu đồng/lượng", "Mua vào: 84,5", "Bán ra: 85"} {
		wantHas(t, out, w)
	}
	for _, bad := range []string{"secret", "Menu Trang chủ", "Bản quyền", "color:red", "<b>"} {
		if strings.Contains(out, bad) {
			t.Errorf("page boilerplate leaked: %q in %s", bad, out)
		}
	}
	wantHas(t, fetchURL(t.Context(), srv.URL+"/plain", 0), "văn bản thuần")
	wantHas(t, fetchURL(t.Context(), srv.URL+"/empty", 0), "không có nội dung")
	wantHas(t, fetchURL(t.Context(), srv.URL+"/missing", 0), "HTTP 500")
	for _, u := range []string{"", "not a url", "file:///etc/passwd", "ftp://x/y", "javascript:alert(1)"} {
		wantHas(t, fetchURL(t.Context(), u, 0), "URL không hợp lệ")
	}
	old := readChunkChars
	readChunkChars = 20
	defer func() { readChunkChars = old }()
	wantHas(t, fetchURL(t.Context(), srv.URL+"/article", 0), "offset=") // long pages are paged
}

func TestWebToolsApprovalAndOffline(t *testing.T) {
	if !needsApproval("fetch_url") || needsApproval("web_search") {
		t.Error("fetch_url must need approval (data could leak via URL); web_search must not")
	}
	srv := webMock(t)
	oldD := ddgURL
	defer func() { ddgURL = oldD }()
	ddgURL = srv.URL + "/html/"
	b := newBot(t)
	b.expect("google giá vàng", "Giá vàng hôm nay", "vnexpress.net")
	b.expect("tin tức bóng đá", "Tuổi Trẻ")
	b.expect("tin tức", "Tin gì")
	b.expect("google", "Tìm gì")
	b.expect("đọc "+srv.URL+"/article", "Vàng tăng mạnh")
	b.expect("mở "+srv.URL+"/plain", "văn bản thuần")
}
