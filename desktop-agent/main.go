package main

import (
	"context"
	"crypto/rand"
	"embed"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
)

//go:embed web
var webFS embed.FS

// Server is the runner of ONE chat session: its project folder, LLM history, pending approvals.
// The App below owns many of them (one per session).
type Server struct {
	cfg       Config
	mu        sync.Mutex
	project   string
	history   []Msg
	pending   map[string]chan bool
	pendInfo  map[string]map[string]any // what the approval card shows (for redrawing after a session switch)
	auto      bool
	busy      bool
	cancel    context.CancelFunc
	forceText bool
	hint      func(string)
}

func (s *Server) isAutoApprove() bool { s.mu.Lock(); defer s.mu.Unlock(); return s.auto }

func main() {
	noBrowser := flag.Bool("no-browser", false, "do not open the browser (service mode: never auto-exits)")
	project := flag.String("project", "", "project folder to open")
	selftest := flag.Bool("selftest", false, "run 3 multi-step goals against the configured LLM and print PASS/FAIL, then exit")
	flag.Parse()

	if dir := appDataDir(); dir != "" {
		_ = os.MkdirAll(dir, 0o755)
		if f, err := os.OpenFile(filepath.Join(dir, "agent.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644); err == nil {
			log.SetOutput(f)
		}
	}

	cfg := loadConfig()
	if *selftest {
		os.Exit(selftestMain(cfg))
	}

	// ---- one copy only: a second launch just brings the running one to the front
	alreadyRunning := func() bool {
		i, ok := runningInstance()
		if !ok {
			return false
		}
		fmt.Println("Chương trình đã chạy sẵn, mở lại tab:", i.URL())
		if *project != "" {
			if err := askRunningToOpen(i, *project); err != nil {
				log.Printf("--project on the running instance: %v", err)
			}
		}
		if !*noBrowser {
			openBrowser(i.URL())
		}
		return true
	}
	if alreadyRunning() {
		return
	}
	ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", cfg.Port))
	if err != nil {
		// maybe another copy that is still starting up (it writes instance.json right after it binds)
		for i := 0; i < 15; i++ {
			time.Sleep(200 * time.Millisecond)
			if alreadyRunning() {
				return
			}
		}
		// the port belongs to some other program: take a free one instead
		if ln, err = net.Listen("tcp", "127.0.0.1:0"); err != nil {
			log.Fatal(err)
		}
	}

	defaultCmdTimeout = cfg.CmdTimeout
	if strings.TrimSpace(cfg.BaseURL) != "" { // small-context LLM: page file reads to fit
		readChunkChars = min(8000, max(1500, cfg.ContextTokens/5))
	}
	if dir := appDataDir(); dir != "" {
		sched = newScheduler(filepath.Join(dir, "tasks.json"), cfg.AlertWebhook)
	} else {
		sched = newScheduler("", cfg.AlertWebhook)
	}
	sched.Start()
	webConf = webCfg{Provider: cfg.SearchProvider, GoogleKey: cfg.GoogleAPIKey, GoogleCX: cfg.GoogleCX, SearxURL: cfg.SearxngURL}
	tokHex := cfg.AccessToken
	if len(tokHex) < 16 { // too short/empty → random per start
		if tokHex != "" {
			log.Printf("access_token too short (need >=16 chars), using a random one")
		}
		tok := make([]byte, 16)
		_, _ = rand.Read(tok)
		tokHex = hex.EncodeToString(tok)
	}
	wsPath := ""
	if dir := appDataDir(); dir != "" {
		wsPath = filepath.Join(dir, "workspace.json")
	}
	app := &App{cfg: cfg, token: tokHex, runners: map[string]*Server{}, ws: loadWorkspace(wsPath)}
	app.lastSeen.Store(time.Now().UnixNano())
	app.ws.mu.Lock()
	app.ws.pickActiveLocked()
	app.ws.mu.Unlock()
	if *project != "" {
		if _, _, err := app.ws.OpenProject(*project); err != nil {
			log.Printf("--project: %v", err)
		}
	}

	app.addr = ln.Addr().String()
	writeInstance(instanceInfo{PID: os.Getpid(), Addr: app.addr, Token: app.token})
	app.cleanup = func() { removeInstance(app.addr) }
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	go func() { <-sig; app.cleanup(); os.Exit(0) }()

	url := fmt.Sprintf("http://%s/?t=%s", app.addr, app.token)
	log.Printf("listening on %s model=%s base=%s pid=%d", app.addr, cfg.Model, cfg.BaseURL, os.Getpid())
	fmt.Println("Open:", url)

	// ---- do not linger invisibly: when the browser tab is gone and nothing needs the program, end it
	if idle := time.Duration(cfg.AutoExitSeconds) * time.Second; !*noBrowser && idle > 0 {
		go func() {
			for {
				time.Sleep(time.Second)
				if app.shouldAutoExit(time.Now(), idle) {
					log.Printf("no browser tab for %v and no work in progress: exiting", idle)
					app.cleanup()
					exitProcess(0)
				}
			}
		}()
	}

	sub, _ := fs.Sub(webFS, "web")
	mux := http.NewServeMux()
	mux.Handle("/", http.FileServerFS(sub))
	app.routes(mux)
	if !*noBrowser {
		go openBrowser(url)
	}
	log.Fatal(http.Serve(ln, mux))
}

func openBrowser(url string) {
	var c *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		c = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	case "darwin":
		c = exec.Command("open", url)
	default:
		c = exec.Command("xdg-open", url)
	}
	_ = c.Start()
}

func osName() string { return runtime.GOOS }

// selftestMain prints to stdout AND to selftest-report.txt (a GUI-subsystem Windows exe has no console).
func selftestMain(cfg Config) int {
	var w io.Writer = os.Stdout
	if f, err := os.Create("selftest-report.txt"); err == nil {
		defer f.Close()
		w = io.MultiWriter(os.Stdout, f)
	}
	return runSelfTest(cfg, w)
}
