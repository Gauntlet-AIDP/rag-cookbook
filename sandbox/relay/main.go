// relay: forwards TCP :8765 to the host claude-helper and nothing else.
package main

import (
	"io"
	"log"
	"net"
	"time"
)

const (
	listenAddr  = ":8765"
	target      = "host.docker.internal:8765"
	maxConns    = 8
	idleTimeout = 330 * time.Second // just over the helper's 300s claude limit
	maxLife     = 6 * time.Minute
)

// deadlineConn re-arms an idle deadline on every read and write, capped by a hard overall deadline.
type deadlineConn struct {
	net.Conn
	hard time.Time
}

func (c deadlineConn) arm() {
	d := time.Now().Add(idleTimeout)
	if d.After(c.hard) {
		d = c.hard
	}
	_ = c.Conn.SetDeadline(d)
}

func (c deadlineConn) Read(p []byte) (int, error) {
	c.arm()
	return c.Conn.Read(p)
}

func (c deadlineConn) Write(p []byte) (int, error) {
	c.arm()
	return c.Conn.Write(p)
}

func (c deadlineConn) CloseWrite() error {
	if tc, ok := c.Conn.(*net.TCPConn); ok {
		return tc.CloseWrite()
	}
	return nil
}

func pipe(dst, src deadlineConn) {
	_, _ = io.Copy(dst, src)
	_ = dst.CloseWrite()
}

func handle(client net.Conn) {
	defer client.Close()
	hard := time.Now().Add(maxLife)
	upstream, err := net.DialTimeout("tcp", target, 5*time.Second)
	if err != nil {
		log.Printf("dial %s: %v", target, err)
		return
	}
	defer upstream.Close()
	c, up := deadlineConn{client, hard}, deadlineConn{upstream, hard}
	done := make(chan struct{}, 2)
	go func() { pipe(up, c); done <- struct{}{} }()
	go func() { pipe(c, up); done <- struct{}{} }()
	<-done
	<-done
}

func main() {
	l, err := net.Listen("tcp", listenAddr)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("relaying %s -> %s", listenAddr, target)
	sem := make(chan struct{}, maxConns)
	for {
		c, err := l.Accept()
		if err != nil {
			log.Fatal(err)
		}
		select {
		case sem <- struct{}{}:
			go func() {
				defer func() { <-sem }()
				handle(c)
			}()
		default:
			log.Printf("rejecting connection: %d already active", maxConns)
			c.Close()
		}
	}
}
