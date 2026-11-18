//go:build !windows

package main

import (
	"context"
	"os/exec"
	"syscall"
)

// newShellCmd runs the command in its own process group so Cancel can kill children too.
func newShellCmd(ctx context.Context, command string) *exec.Cmd {
	c := exec.CommandContext(ctx, "sh", "-c", command)
	c.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	c.Cancel = func() error { return syscall.Kill(-c.Process.Pid, syscall.SIGKILL) }
	return c
}
