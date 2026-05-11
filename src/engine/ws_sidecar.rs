use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct WsSidecarConfig {
    pub bind_addr: String,
}

#[derive(Debug)]
pub struct WsSidecarHandle {
    pub events_tx: Sender<String>,
    #[cfg(test)]
    pub bound_addr: String,
}

pub fn start_ws_sidecar(config: WsSidecarConfig) -> Result<WsSidecarHandle, String> {
    let listener = TcpListener::bind(&config.bind_addr)
        .map_err(|e| format!("ws sidecar bind {}: {e}", config.bind_addr))?;
    #[cfg(test)]
    let bound_addr = listener
        .local_addr()
        .map_err(|e| format!("ws sidecar local_addr: {e}"))?
        .to_string();
    listener
        .set_nonblocking(true)
        .map_err(|e| format!("ws sidecar set_nonblocking: {e}"))?;

    let (events_tx, events_rx) = mpsc::channel::<String>();
    std::thread::spawn(move || run_ws_sidecar(listener, events_rx));

    Ok(WsSidecarHandle {
        events_tx,
        #[cfg(test)]
        bound_addr,
    })
}

fn run_ws_sidecar(listener: TcpListener, events_rx: Receiver<String>) {
    let mut clients: Vec<TcpStream> = Vec::new();
    let mut last_keepalive = Instant::now();

    loop {
        match listener.accept() {
            Ok((mut stream, _)) => {
                let _ = stream.set_nodelay(true);
                // Handshakes can get delayed under CPU contention; don't fail upgrades too eagerly.
                let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
                if let Ok(true) = maybe_upgrade_to_websocket(&mut stream) {
                    clients.push(stream);
                } else {
                    let _ = write_http_response(
                        &mut stream,
                        400,
                        "Bad Request",
                        "text/plain",
                        b"bad websocket upgrade\n",
                    );
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(_) => {}
        }

        let mut any_event = false;
        loop {
            match events_rx.try_recv() {
                Ok(line) => {
                    any_event = true;
                    broadcast_ws_text(&mut clients, &line);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }

        if !any_event && last_keepalive.elapsed() >= Duration::from_secs(15) {
            broadcast_ws_ping(&mut clients);
            last_keepalive = Instant::now();
        }

        std::thread::sleep(Duration::from_millis(20));
    }
}

fn maybe_upgrade_to_websocket(stream: &mut TcpStream) -> Result<bool, std::io::Error> {
    let Some(raw) = read_http_request(stream, 8 * 1024)? else { return Ok(false) };

    let (request_line, headers) = parse_http_request(&raw);
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("");
    if method != "GET" || path != "/ws" {
        return Ok(false);
    }

    let upgrade = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("Upgrade"))
        .map(|(_, v)| v.to_ascii_lowercase())
        .unwrap_or_default();
    if upgrade != "websocket" {
        return Ok(false);
    }

    let connection = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("Connection"))
        .map(|(_, v)| v.to_ascii_lowercase())
        .unwrap_or_default();
    if !connection.split(',').any(|t| t.trim() == "upgrade") {
        return Ok(false);
    }

    let key = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("Sec-WebSocket-Key"))
        .map(|(_, v)| v.trim().to_string())
        .unwrap_or_default();
    if key.is_empty() {
        return Ok(false);
    }

    let accept = websocket_accept_key(&key);
    let response = format!(
        "HTTP/1.1 101 Switching Protocols\r\n\
Upgrade: websocket\r\n\
Connection: Upgrade\r\n\
Sec-WebSocket-Accept: {accept}\r\n\
Access-Control-Allow-Origin: *\r\n\
\r\n"
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()?;
    Ok(true)
}

fn read_http_request(
    stream: &mut TcpStream,
    max_bytes: usize,
) -> Result<Option<Vec<u8>>, std::io::Error> {
    let mut buf = Vec::<u8>::new();
    let mut tmp = [0u8; 1024];
    loop {
        let n = match stream.read(&mut tmp) {
            Ok(0) => return Ok(None),
            Ok(n) => n,
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                return Ok(None);
            }
            Err(err) => return Err(err),
        };
        buf.extend_from_slice(&tmp[..n]);
        if buf.len() > max_bytes {
            return Ok(None);
        }
        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
            return Ok(Some(buf));
        }
    }
}

fn parse_http_request(raw: &[u8]) -> (String, Vec<(String, String)>) {
    let text = String::from_utf8_lossy(raw);
    let mut lines = text.split("\r\n");
    let request_line = lines.next().unwrap_or("").to_string();
    let mut headers = Vec::<(String, String)>::new();
    for line in lines {
        if line.is_empty() {
            break;
        }
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        headers.push((k.trim().to_string(), v.trim().to_string()));
    }
    (request_line, headers)
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

fn broadcast_ws_text(clients: &mut Vec<TcpStream>, text: &str) {
    let payload = text.as_bytes();
    let frame = build_ws_frame(0x1, payload);
    write_to_clients(clients, &frame);
}

fn broadcast_ws_ping(clients: &mut Vec<TcpStream>) {
    let frame = build_ws_frame(0x9, b"");
    write_to_clients(clients, &frame);
}

fn write_to_clients(clients: &mut Vec<TcpStream>, bytes: &[u8]) {
    let mut i = 0usize;
    while i < clients.len() {
        let ok = clients[i].write_all(bytes).is_ok() && clients[i].flush().is_ok();
        if ok {
            i += 1;
        } else {
            let _ = clients.swap_remove(i);
        }
    }
}

fn build_ws_frame(opcode: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::<u8>::with_capacity(payload.len() + 16);
    out.push(0x80 | (opcode & 0x0F)); // FIN=1
    if payload.len() <= 125 {
        out.push(payload.len() as u8);
    } else if payload.len() <= 0xFFFF {
        out.push(126);
        out.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    } else {
        out.push(127);
        out.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    }
    out.extend_from_slice(payload);
    out
}

fn websocket_accept_key(client_key: &str) -> String {
    const GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    let mut bytes = Vec::<u8>::new();
    bytes.extend_from_slice(client_key.trim().as_bytes());
    bytes.extend_from_slice(GUID.as_bytes());
    let digest = sha1(&bytes);
    base64_encode(&digest)
}

fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let b0 = bytes[i];
        let b1 = if i + 1 < bytes.len() { bytes[i + 1] } else { 0 };
        let b2 = if i + 2 < bytes.len() { bytes[i + 2] } else { 0 };
        let n = (u32::from(b0) << 16) | (u32::from(b1) << 8) | u32::from(b2);
        let c0 = TABLE[((n >> 18) & 0x3F) as usize] as char;
        let c1 = TABLE[((n >> 12) & 0x3F) as usize] as char;
        let c2 = TABLE[((n >> 6) & 0x3F) as usize] as char;
        let c3 = TABLE[(n & 0x3F) as usize] as char;
        out.push(c0);
        out.push(c1);
        if i + 1 < bytes.len() {
            out.push(c2);
        } else {
            out.push('=');
        }
        if i + 2 < bytes.len() {
            out.push(c3);
        } else {
            out.push('=');
        }
        i += 3;
    }
    out
}

fn sha1(input: &[u8]) -> [u8; 20] {
    let mut h0: u32 = 0x67452301;
    let mut h1: u32 = 0xEFCDAB89;
    let mut h2: u32 = 0x98BADCFE;
    let mut h3: u32 = 0x10325476;
    let mut h4: u32 = 0xC3D2E1F0;

    let mut msg = input.to_vec();
    let bit_len = (msg.len() as u64) * 8;
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    let mut w = [0u32; 80];
    for chunk in msg.chunks_exact(64) {
        for (i, word) in w.iter_mut().take(16).enumerate() {
            let j = i * 4;
            *word = u32::from_be_bytes([chunk[j], chunk[j + 1], chunk[j + 2], chunk[j + 3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let mut a = h0;
        let mut b = h1;
        let mut c = h2;
        let mut d = h3;
        let mut e = h4;

        for (i, word) in w.iter().enumerate() {
            let (f, k) = if i < 20 {
                ((b & c) | ((!b) & d), 0x5A827999)
            } else if i < 40 {
                (b ^ c ^ d, 0x6ED9EBA1)
            } else if i < 60 {
                ((b & c) | (b & d) | (c & d), 0x8F1BBCDC)
            } else {
                (b ^ c ^ d, 0xCA62C1D6)
            };

            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut out = [0u8; 20];
    out[0..4].copy_from_slice(&h0.to_be_bytes());
    out[4..8].copy_from_slice(&h1.to_be_bytes());
    out[8..12].copy_from_slice(&h2.to_be_bytes());
    out[12..16].copy_from_slice(&h3.to_be_bytes());
    out[16..20].copy_from_slice(&h4.to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::time::Duration;

    fn read_until_double_crlf(stream: &mut TcpStream, max_bytes: usize) -> Vec<u8> {
        let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
        let mut buf = Vec::<u8>::new();
        let mut tmp = [0u8; 1024];
        while buf.len() < max_bytes {
            let n = stream.read(&mut tmp).expect("read");
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        buf
    }

    fn websocket_upgrade(stream: &mut TcpStream, key: &str) -> String {
        let req = format!(
            "GET /ws HTTP/1.1\r\n\
Host: localhost\r\n\
Upgrade: websocket\r\n\
Connection: Upgrade\r\n\
Sec-WebSocket-Key: {key}\r\n\
Sec-WebSocket-Version: 13\r\n\
\r\n"
        );
        stream.write_all(req.as_bytes()).expect("write");
        stream.flush().expect("flush");

        let raw = read_until_double_crlf(stream, 16 * 1024);
        String::from_utf8_lossy(&raw).to_string()
    }

    #[test]
    fn websocket_upgrade_returns_101() {
        let handle = start_ws_sidecar(WsSidecarConfig {
            bind_addr: "127.0.0.1:0".to_string(),
        })
        .expect("start_ws_sidecar");

        let mut stream = TcpStream::connect(&handle.bound_addr).expect("connect");
        let resp = websocket_upgrade(&mut stream, "dGhlIHNhbXBsZSBub25jZQ==");

        assert!(resp.contains("HTTP/1.1 101 Switching Protocols"));
        assert!(resp.contains("Upgrade: websocket"));
        assert!(resp.contains("Connection: Upgrade"));
        assert!(resp.contains("Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo="));
        drop(handle);
    }

    #[test]
    fn websocket_broadcast_sends_text_frame() {
        let handle = start_ws_sidecar(WsSidecarConfig {
            bind_addr: "127.0.0.1:0".to_string(),
        })
        .expect("start_ws_sidecar");

        let mut stream = TcpStream::connect(&handle.bound_addr).expect("connect");
        let resp = websocket_upgrade(&mut stream, "dGhlIHNhbXBsZSBub25jZQ==");
        assert!(resp.contains("101 Switching Protocols"));

        let msg = "hello-ws";
        handle.events_tx.send(msg.to_string()).expect("send");

        // Read a single server->client text frame (unmasked).
        let _ = stream.set_read_timeout(Some(Duration::from_millis(1200)));
        let mut head = [0u8; 2];
        stream.read_exact(&mut head).expect("read head");
        assert_eq!(head[0], 0x81); // FIN=1 + opcode=1 (text)
        assert_eq!(head[1] & 0x80, 0); // not masked

        let mut len = (head[1] & 0x7F) as usize;
        if len == 126 {
            let mut ext = [0u8; 2];
            stream.read_exact(&mut ext).expect("read ext");
            len = u16::from_be_bytes(ext) as usize;
        } else if len == 127 {
            let mut ext = [0u8; 8];
            stream.read_exact(&mut ext).expect("read ext");
            len = u64::from_be_bytes(ext) as usize;
        }

        let mut payload = vec![0u8; len];
        stream.read_exact(&mut payload).expect("read payload");
        assert_eq!(std::str::from_utf8(&payload).unwrap(), msg);
        drop(handle);
    }
}

#[cfg(test)]
mod accept_key_tests {
    use super::websocket_accept_key;

    #[test]
    fn websocket_accept_key_matches_rfc_example() {
        let key = "dGhlIHNhbXBsZSBub25jZQ==";
        let accept = websocket_accept_key(key);
        assert_eq!(accept, "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    }
}
