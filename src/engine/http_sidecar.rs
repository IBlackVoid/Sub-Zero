use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct HttpSidecarConfig {
    pub bind_addr: String,
}

#[derive(Debug)]
pub struct HttpSidecarHandle {
    pub events_tx: Sender<String>,
    #[cfg(test)]
    pub bound_addr: String,
}

pub fn start_http_sidecar(config: HttpSidecarConfig) -> Result<HttpSidecarHandle, String> {
    let listener = TcpListener::bind(&config.bind_addr)
        .map_err(|e| format!("http sidecar bind {}: {e}", config.bind_addr))?;
    #[cfg(test)]
    let bound_addr = listener
        .local_addr()
        .map_err(|e| format!("http sidecar local_addr: {e}"))?
        .to_string();
    listener
        .set_nonblocking(true)
        .map_err(|e| format!("http sidecar set_nonblocking: {e}"))?;

    let (events_tx, events_rx) = mpsc::channel::<String>();
    std::thread::spawn(move || run_http_sidecar(listener, events_rx));

    Ok(HttpSidecarHandle {
        events_tx,
        #[cfg(test)]
        bound_addr,
    })
}

fn run_http_sidecar(listener: TcpListener, events_rx: Receiver<String>) {
    let mut clients: Vec<TcpStream> = Vec::new();
    let mut last_keepalive = Instant::now();

    loop {
        match listener.accept() {
            Ok((mut stream, _)) => {
                if let Ok(Some(kind)) = classify_request(&mut stream) {
                    match kind {
                        RequestKind::Events => {
                            let _ = write_sse_headers(&mut stream);
                            let _ = stream.set_write_timeout(Some(Duration::from_millis(250)));
                            let _ = stream.set_nodelay(true);
                            clients.push(stream);
                        }
                        RequestKind::Health => {
                            let _ = write_http_response(
                                &mut stream,
                                200,
                                "OK",
                                "application/json",
                                b"{\"ok\":true}\n",
                            );
                        }
                        RequestKind::NotFound => {
                            let _ = write_http_response(
                                &mut stream,
                                404,
                                "Not Found",
                                "text/plain",
                                b"not found\n",
                            );
                        }
                    }
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(_) => {
                // Listener failure: nothing actionable here without logging deps.
            }
        }

        let mut any_event = false;
        loop {
            match events_rx.try_recv() {
                Ok(line) => {
                    any_event = true;
                    broadcast_sse(&mut clients, &line);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    // Main thread dropped the sender: exit.
                    return;
                }
            }
        }

        // Keepalive comment lines so proxies/clients don't time out.
        if !any_event && last_keepalive.elapsed() >= Duration::from_secs(15) {
            broadcast_sse_comment(&mut clients, b": keepalive\n\n");
            last_keepalive = Instant::now();
        }

        std::thread::sleep(Duration::from_millis(20));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestKind {
    Events,
    Health,
    NotFound,
}

fn classify_request(stream: &mut TcpStream) -> Result<Option<RequestKind>, std::io::Error> {
    // Players/UIs may connect and then send immediately; give them a little headroom so
    // we don't accidentally drop the socket during scheduling jitter.
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    let mut buf = [0u8; 4096];
    let n = match stream.read(&mut buf) {
        Ok(0) => return Ok(None),
        Ok(n) => n,
        Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => return Ok(None),
        Err(err) => return Err(err),
    };
    let raw = String::from_utf8_lossy(&buf[..n]);
    let Some(line) = raw.lines().next() else {
        return Ok(Some(RequestKind::NotFound));
    };
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("/");
    if method != "GET" {
        return Ok(Some(RequestKind::NotFound));
    }
    if path == "/events" {
        return Ok(Some(RequestKind::Events));
    }
    if path == "/health" || path == "/" {
        return Ok(Some(RequestKind::Health));
    }
    Ok(Some(RequestKind::NotFound))
}

fn write_sse_headers(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(b"HTTP/1.1 200 OK\r\n")?;
    stream.write_all(b"Content-Type: text/event-stream\r\n")?;
    stream.write_all(b"Cache-Control: no-cache\r\n")?;
    stream.write_all(b"Connection: keep-alive\r\n")?;
    stream.write_all(b"Access-Control-Allow-Origin: *\r\n")?;
    stream.write_all(b"\r\n")?;
    stream.flush()
}

fn write_http_response(
    stream: &mut TcpStream,
    code: u16,
    reason: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn broadcast_sse(clients: &mut Vec<TcpStream>, json_line: &str) {
    let mut payload = Vec::with_capacity(json_line.len() + 16);
    payload.extend_from_slice(b"data: ");
    payload.extend_from_slice(json_line.as_bytes());
    payload.extend_from_slice(b"\n\n");
    broadcast_sse_comment(clients, &payload);
}

fn broadcast_sse_comment(clients: &mut Vec<TcpStream>, bytes: &[u8]) {
    let mut i = 0usize;
    while i < clients.len() {
        let write_ok = clients[i].write_all(bytes).is_ok() && clients[i].flush().is_ok();
        if write_ok {
            i += 1;
        } else {
            let _ = clients.swap_remove(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::time::Duration;

    fn read_until_contains(stream: &mut TcpStream, needle: &str, max_wait_ms: u64) -> String {
        let _ = stream.set_read_timeout(Some(Duration::from_millis(150)));
        let started = Instant::now();
        let mut out = Vec::<u8>::new();
        let mut buf = [0u8; 2048];
        while started.elapsed() < Duration::from_millis(max_wait_ms) {
            match stream.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => out.extend_from_slice(&buf[..n]),
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(err) if err.kind() == std::io::ErrorKind::TimedOut => {}
                Err(_) => break,
            }
            if String::from_utf8_lossy(&out).contains(needle) {
                break;
            }
        }
        String::from_utf8_lossy(&out).to_string()
    }

    #[test]
    fn health_endpoint_returns_ok_json() {
        let handle = start_http_sidecar(HttpSidecarConfig {
            bind_addr: "127.0.0.1:0".to_string(),
        })
        .expect("start_http_sidecar");

        let mut stream = TcpStream::connect(&handle.bound_addr).expect("connect");
        stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .expect("write request");
        stream.flush().expect("flush");

        let text = read_until_contains(&mut stream, "{\"ok\":true}", 1500);
        assert!(text.contains("HTTP/1.1 200 OK"), "{text}");
        assert!(text.contains("{\"ok\":true}"), "{text}");
        drop(handle);
    }

    #[test]
    fn events_endpoint_emits_sse_data_lines() {
        let handle = start_http_sidecar(HttpSidecarConfig {
            bind_addr: "127.0.0.1:0".to_string(),
        })
        .expect("start_http_sidecar");

        let mut stream = TcpStream::connect(&handle.bound_addr).expect("connect");
        stream
            .write_all(b"GET /events HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .expect("write request");
        stream.flush().expect("flush");

        // Wait for headers.
        let header_text = read_until_contains(&mut stream, "text/event-stream", 1500);
        assert!(header_text.contains("HTTP/1.1 200 OK"), "{header_text}");
        assert!(header_text.contains("text/event-stream"), "{header_text}");

        handle
            .events_tx
            .send("{\"event\":\"test\"}".to_string())
            .expect("send event");

        // Wait for at least one SSE data line.
        let payload_text = read_until_contains(&mut stream, "data: {\"event\":\"test\"}", 1500);
        assert!(
            payload_text.contains("data: {\"event\":\"test\"}"),
            "{payload_text}"
        );
        drop(handle);
    }
}
