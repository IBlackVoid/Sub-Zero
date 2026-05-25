//! JSONL animation loader for the cells format produced by
//! `Pixel-Ripper_V16/ascii_engine.exe --output-format cells`.
//!
//! Wire format (one frame per JSONL line):
//!
//! ```json
//! {"frame":N,"delay_ms":D,"width":W,"height":H,
//!  "rows":[[ [ch,r,g,b], ... ], ...]}
//! ```
//!
//! The loader is deliberately tolerant: malformed lines are skipped
//! with a warning, missing colours default to white, and out-of-range
//! cells are clamped at the row width seen in the header.

use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Clone, Copy)]
pub struct Cell {
    pub ch: char,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub delay_ms: u32,
    pub width: u16,
    pub height: u16,
    /// Row-major: `cells[row * width + col]`.
    pub cells: Vec<Cell>,
}

#[derive(Debug, Clone)]
pub struct Animation {
    pub frames: Vec<Frame>,
    pub width: u16,
    /// Total wall-time duration of one loop, summed from per-frame
    /// `delay_ms` (or 80 ms default for missing values). Used by the
    /// TUI to keep audio and visual playheads in sync.
    pub total_duration_ms: u64,
    /// Sibling audio file path, if one exists next to the JSONL.
    /// Convention: same stem, `.wav` extension. Set on load.
    pub audio_path: Option<std::path::PathBuf>,
}

/// Raw shape we deserialise from each JSONL line. Compact tuple cells
/// are decoded as `Vec<serde_json::Value>` then walked manually so we
/// can tolerate the occasional missing colour channel without aborting
/// the whole frame.
#[derive(Debug, Deserialize)]
struct RawFrame {
    #[serde(default)]
    delay_ms: u32,
    width: u16,
    height: u16,
    rows: Vec<Vec<serde_json::Value>>,
}

impl Animation {
    /// Load a JSONL animation from disk. Returns an error only if the
    /// file is missing or contains zero usable frames. Auto-detects a
    /// sibling `.wav` audio track and stores its path.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, AnimationError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(AnimationError::Io)?;
        let mut frames = Vec::new();
        let mut width = 0u16;

        for (line_no, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let raw: RawFrame = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!(
                        "sub-zero-tui: skipping line {} of {}: {}",
                        line_no + 1,
                        path.display(),
                        e
                    );
                    continue;
                }
            };
            let frame = decode_frame(raw);
            if width == 0 {
                width = frame.width;
            }
            frames.push(frame);
        }

        if frames.is_empty() {
            return Err(AnimationError::Empty(path.to_path_buf()));
        }

        // Sum effective per-frame delays for sync. Frames with
        // `delay_ms == 0` use the renderer's 80 ms default.
        let total_duration_ms = frames
            .iter()
            .map(|f| {
                if f.delay_ms == 0 {
                    80
                } else {
                    f.delay_ms as u64
                }
            })
            .sum::<u64>();

        let audio_path = path.with_extension("wav");
        let audio_path = if audio_path.is_file() {
            Some(audio_path)
        } else {
            None
        };

        Ok(Self {
            frames,
            width,
            total_duration_ms,
            audio_path,
        })
    }
}

fn decode_frame(raw: RawFrame) -> Frame {
    let total = (raw.width as usize) * (raw.height as usize);
    let mut cells = Vec::with_capacity(total);
    for row in raw.rows {
        let mut col = 0u16;
        for cell_v in row {
            if col >= raw.width {
                break;
            }
            cells.push(decode_cell(&cell_v));
            col += 1;
        }
        // Pad short rows with blanks so the buffer stays rectangular.
        while col < raw.width {
            cells.push(Cell {
                ch: ' ',
                r: 0,
                g: 0,
                b: 0,
            });
            col += 1;
        }
    }
    // Pad missing rows so cells.len() == width * height.
    while cells.len() < total {
        cells.push(Cell {
            ch: ' ',
            r: 0,
            g: 0,
            b: 0,
        });
    }
    Frame {
        delay_ms: raw.delay_ms,
        width: raw.width,
        height: raw.height,
        cells,
    }
}

fn decode_cell(v: &serde_json::Value) -> Cell {
    // Expected shape: [string, u8, u8, u8].
    let arr = match v.as_array() {
        Some(a) => a,
        None => {
            return Cell {
                ch: ' ',
                r: 255,
                g: 255,
                b: 255,
            }
        }
    };
    let ch = arr
        .first()
        .and_then(|s| s.as_str())
        .and_then(|s| s.chars().next())
        .unwrap_or(' ');
    let r = arr
        .get(1)
        .and_then(|v| v.as_u64())
        .map(|v| v as u8)
        .unwrap_or(255);
    let g = arr
        .get(2)
        .and_then(|v| v.as_u64())
        .map(|v| v as u8)
        .unwrap_or(255);
    let b = arr
        .get(3)
        .and_then(|v| v.as_u64())
        .map(|v| v as u8)
        .unwrap_or(255);
    Cell { ch, r, g, b }
}

#[derive(Debug)]
pub enum AnimationError {
    Io(std::io::Error),
    Empty(std::path::PathBuf),
}

impl std::fmt::Display for AnimationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Empty(p) => write!(f, "no usable frames in {}", p.display()),
        }
    }
}

impl std::error::Error for AnimationError {}
