use serde_json::Value;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct EventSink {
    enabled: bool,
    file: Option<Arc<Mutex<std::fs::File>>>,
    http_events: Option<Sender<String>>,
    ws_events: Option<Sender<String>>,
}

impl EventSink {
    pub fn new(
        enabled: bool,
        file_path: Option<&Path>,
        http_events: Option<Sender<String>>,
        ws_events: Option<Sender<String>>,
    ) -> Result<Self, String> {
        let file = file_path
            .map(|path| {
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .map_err(|e| format!("{}: {e}", path.display()))
                    .map(|f| Arc::new(Mutex::new(f)))
            })
            .transpose()?;

        Ok(Self {
            enabled,
            file,
            http_events,
            ws_events,
        })
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn emit(&self, payload: &Value) {
        if !self.enabled {
            return;
        }

        let Ok(line) = serde_json::to_string(payload) else {
            return;
        };

        println!("{line}");

        if let Some(tx) = self.http_events.as_ref() {
            // Non-fatal: sidecar is best-effort observability.
            let _ = tx.send(line.clone());
        }
        if let Some(tx) = self.ws_events.as_ref() {
            let _ = tx.send(line.clone());
        }

        let Some(file) = self.file.as_ref() else {
            return;
        };
        let Ok(mut guard) = file.lock() else {
            return;
        };
        let _ = guard.write_all(line.as_bytes());
        let _ = guard.write_all(b"\n");
        let _ = guard.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn temp_path(stem: &str) -> std::path::PathBuf {
        let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("voidex_{stem}_{n}.jsonl"))
    }

    #[test]
    fn events_file_appends_lines() {
        let path = temp_path("events_append");
        std::fs::write(&path, "first\n").expect("write seed file");

        let sink = EventSink::new(true, Some(&path), None, None).expect("EventSink::new");
        sink.emit(&json!({
            "event": "test",
            "n": 1,
        }));
        drop(sink);

        let content = std::fs::read_to_string(&path).expect("read file");
        assert!(content.starts_with("first\n"));
        assert!(content.contains("\"event\":\"test\""));
        let _ = std::fs::remove_file(&path);
    }
}
