//go:build windows

package main

import (
	"context"
	"os/exec"
	"strconv"
	"syscall"
)

const createNoWindow = 0x08000000

// newShellCmd builds a cmd.exe invocation.
//   - CmdLine is set by hand: Go's default argument quoting breaks cmd.exe when the
//     command contains quotes (e.g. dir "C:\Program Files").
//   - chcp 65001 makes output UTF-8 so Vietnamese text is not garbled.
//   - Cancel kills the whole process tree (otherwise e.g. `ping -t` outlives cmd.exe).
func newShellCmd(ctx context.Context, command string) *exec.Cmd {
	c := exec.CommandContext(ctx, "cmd.exe")
	c.SysProcAttr = &syscall.SysProcAttr{
		HideWindow:    true,
		CreationFlags: createNoWindow,
		CmdLine:       `cmd.exe /S /C "chcp 65001>nul & ` + command + `"`,
	}
	c.Cancel = func() error {
		k := exec.Command("taskkill", "/T", "/F", "/PID", strconv.Itoa(c.Process.Pid))
		k.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: createNoWindow}
		_ = k.Run()
		return c.Process.Kill()
	}
	return c
}
