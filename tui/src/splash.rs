use ratatui::{
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};
use std::time::{Duration, Instant};

/// Total splash animation duration. Five phases:
///   0.0–0.4s  rift opens (point → horizontal line)
///   0.4–1.6s  letters emerge from the rift, V and D glow
///   1.6–2.4s  hold: name pulses, D void breathes
///   2.4–3.2s  void eats: D expands, swallows letters inward
///   3.2–3.6s  collapse to point → fade to black → UI appears
pub const SPLASH_DURATION: Duration = Duration::from_millis(3_600);

// ── The VoiDex ASCII logo ────────────────────────────────────────────────────
//
// The V has waveform edges (≈ and ~ characters). The D is hollow (outline
// only, interior is void/black). These are the two "special" characters
// in the VoiDex brand identity.

// Individual letter art reserved for future per-character animation.
// Currently the full WORDMARK is used for all phases.

// Waveform characters that flicker on V's edges.
const WAVE_CHARS: &[char] = &['≈', '~', '∿', '≋', '∼'];

// The full assembled wordmark for the "hold" phase.
const WORDMARK: &[&str] = &[
    "██    ██         ██ ████████               ",
    " ██  ██  █████  ██  ██     ██  █████  ██  ██",
    " ██  ██ ██   ██ ██  ██      █ ██   ██  ████ ",
    "  ████  ██   ██ ██  ██      █ ██████   ████ ",
    "   ██    █████  ██  ████████   █████  ██  ██",
];

const TAGLINE: &str = "offline subtitle engine · nothing leaves your machine";

/// Phase boundaries as fractions of SPLASH_DURATION.
const PHASE_RIFT_END: f32 = 0.11;       // 0.0 – 0.11
const PHASE_EMERGE_END: f32 = 0.44;     // 0.11 – 0.44
const PHASE_HOLD_END: f32 = 0.67;       // 0.44 – 0.67
const PHASE_SWALLOW_END: f32 = 0.89;    // 0.67 – 0.89
// 0.89 – 1.0 = collapse to black

pub fn render(f: &mut Frame, area: Rect, started_at: Instant, accent: Color) {
    let elapsed = started_at.elapsed();
    let frac = (elapsed.as_secs_f32() / SPLASH_DURATION.as_secs_f32()).clamp(0.0, 1.0);

    if frac < PHASE_RIFT_END {
        render_rift(f, area, frac / PHASE_RIFT_END, accent);
    } else if frac < PHASE_EMERGE_END {
        let t = (frac - PHASE_RIFT_END) / (PHASE_EMERGE_END - PHASE_RIFT_END);
        render_emerge(f, area, t, accent, elapsed);
    } else if frac < PHASE_HOLD_END {
        let t = (frac - PHASE_EMERGE_END) / (PHASE_HOLD_END - PHASE_EMERGE_END);
        render_hold(f, area, t, accent, elapsed);
    } else if frac < PHASE_SWALLOW_END {
        let t = (frac - PHASE_HOLD_END) / (PHASE_SWALLOW_END - PHASE_HOLD_END);
        render_swallow(f, area, t, accent);
    } else {
        let t = (frac - PHASE_SWALLOW_END) / (1.0 - PHASE_SWALLOW_END);
        render_collapse(f, area, t);
    }
}

// ── Phase 1: Rift opens ──────────────────────────────────────────────────────

fn render_rift(f: &mut Frame, area: Rect, t: f32, accent: Color) {
    // A horizontal line of light expanding from center.
    let mid_y = area.y + area.height / 2;
    let max_half_width = (area.width / 3).min(30) as f32;
    let half_w = (t * max_half_width) as u16;
    let mid_x = area.x + area.width / 2;

    let start_x = mid_x.saturating_sub(half_w);
    let end_x = (mid_x + half_w).min(area.x + area.width);
    let width = end_x.saturating_sub(start_x);

    if width == 0 || mid_y >= area.y + area.height {
        return;
    }

    let intensity = (t * 255.0) as u8;
    let color = scale_color(accent, intensity);
    let rift_char = if t < 0.3 { '·' } else if t < 0.6 { '─' } else { '━' };
    let line_str: String = std::iter::repeat(rift_char).take(width as usize).collect();

    let line = Line::from(Span::styled(
        line_str,
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    ));
    let rect = Rect { x: start_x, y: mid_y, width, height: 1 };
    f.render_widget(Paragraph::new(vec![line]).alignment(Alignment::Center), rect);
}

// ── Phase 2: Letters emerge ──────────────────────────────────────────────────

fn render_emerge(f: &mut Frame, area: Rect, t: f32, accent: Color, elapsed: Duration) {
    // Letters reveal left-to-right. V and D get special treatment.
    let wordmark = WORDMARK;
    let wm_height = wordmark.len() as u16;
    let wm_width = wordmark.iter().map(|r| r.len()).max().unwrap_or(0) as u16;

    if wm_height == 0 || area.height < wm_height + 4 {
        return;
    }

    let pad_y = area.height.saturating_sub(wm_height + 2) / 2;
    let pad_x = area.width.saturating_sub(wm_width) / 2;

    // How many columns to reveal (left to right).
    let reveal_cols = (t * wm_width as f32) as usize;

    let intensity = ((t * 1.5).min(1.0) * 255.0) as u8;
    let base_color = scale_color(Color::Rgb(0xe5, 0xe5, 0xe5), intensity);

    // V accent: columns 0-8, D accent: columns ~28-37 (approximate positions).
    let v_range = 0..9usize;
    let d_range = 28..38usize;

    let wave_idx = (elapsed.as_millis() / 80) as usize;

    let mut lines: Vec<Line> = Vec::with_capacity(wm_height as usize + 4);

    // Waveform on V (left side) — small wave chars above the logo.
    if t > 0.3 {
        let wave_intensity = (((t - 0.3) / 0.7) * 255.0) as u8;
        let wave_color = scale_color(accent, wave_intensity);
        let wave: String = (0..3)
            .map(|i| WAVE_CHARS[(wave_idx + i) % WAVE_CHARS.len()])
            .collect();
        let padding = " ".repeat(pad_x.saturating_sub(4) as usize);
        lines.push(Line::from(Span::styled(
            format!("{padding}  {wave}"),
            Style::default().fg(wave_color),
        )));
    } else {
        lines.push(Line::raw(""));
    }

    for row_str in wordmark {
        let mut spans: Vec<Span> = Vec::new();
        let padding = " ".repeat(pad_x as usize);
        spans.push(Span::raw(padding));

        let chars: Vec<char> = row_str.chars().collect();
        for (col, &ch) in chars.iter().enumerate() {
            if col >= reveal_cols {
                spans.push(Span::raw(" "));
                continue;
            }
            let color = if v_range.contains(&col) {
                // V gets accent color with waveform shimmer.
                let shimmer = ((elapsed.as_millis() as f32 / 120.0 + col as f32).sin() * 0.3 + 0.7) as f32;
                scale_color(accent, (shimmer * intensity as f32) as u8)
            } else if d_range.contains(&col) {
                // D gets dimmed interior (void effect).
                if ch == ' ' {
                    Color::Reset // Interior of D = pure void/black
                } else {
                    base_color
                }
            } else {
                base_color
            };

            let style = if ch == ' ' && d_range.contains(&col) {
                Style::default() // void inside D
            } else {
                Style::default().fg(color).add_modifier(Modifier::BOLD)
            };
            spans.push(Span::styled(ch.to_string(), style));
        }
        lines.push(Line::from(spans));
    }

    // Tagline fades in during last 40% of emerge.
    if t > 0.6 {
        let tag_intensity = (((t - 0.6) / 0.4) * 180.0) as u8;
        let tag_color = scale_color(Color::Rgb(0x80, 0x80, 0x80), tag_intensity);
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled(
            TAGLINE,
            Style::default().fg(tag_color),
        )));
    }

    let body_h = lines.len() as u16;
    let inner = Rect {
        x: area.x,
        y: area.y + pad_y,
        width: area.width,
        height: body_h.min(area.height.saturating_sub(pad_y)),
    };
    f.render_widget(Paragraph::new(lines).alignment(Alignment::Left), inner);
}

// ── Phase 3: Hold + pulse ────────────────────────────────────────────────────

fn render_hold(f: &mut Frame, area: Rect, _t: f32, accent: Color, elapsed: Duration) {
    // Full wordmark visible, V shimmers, D breathes (pulses slightly).
    let wordmark = WORDMARK;
    let wm_height = wordmark.len() as u16;
    let wm_width = wordmark.iter().map(|r| r.len()).max().unwrap_or(0) as u16;

    let pad_y = area.height.saturating_sub(wm_height + 4) / 2;
    let pad_x = area.width.saturating_sub(wm_width) / 2;

    let v_range = 0..9usize;
    let d_range = 28..38usize;

    // D breathes: its outline intensity pulses.
    let breathe = ((elapsed.as_millis() as f32 / 300.0).sin() * 0.15 + 0.85) as f32;
    let base_intensity = (breathe * 255.0) as u8;
    let base_color = scale_color(Color::Rgb(0xe5, 0xe5, 0xe5), base_intensity);

    let wave_idx = (elapsed.as_millis() / 80) as usize;

    let mut lines: Vec<Line> = Vec::with_capacity(wm_height as usize + 4);

    // Waveform above V.
    let wave_color = accent;
    let wave: String = (0..4)
        .map(|i| WAVE_CHARS[(wave_idx + i) % WAVE_CHARS.len()])
        .collect();
    let padding = " ".repeat(pad_x.saturating_sub(4) as usize);
    lines.push(Line::from(Span::styled(
        format!("{padding}  {wave}"),
        Style::default().fg(wave_color),
    )));

    for row_str in wordmark {
        let mut spans: Vec<Span> = Vec::new();
        spans.push(Span::raw(" ".repeat(pad_x as usize)));

        let chars: Vec<char> = row_str.chars().collect();
        for (col, &ch) in chars.iter().enumerate() {
            let color = if v_range.contains(&col) {
                let shimmer = ((elapsed.as_millis() as f32 / 100.0 + col as f32 * 0.5).sin() * 0.3 + 0.7) as f32;
                scale_color(accent, (shimmer * 255.0) as u8)
            } else if d_range.contains(&col) && ch == ' ' {
                Color::Reset
            } else {
                base_color
            };
            spans.push(Span::styled(
                ch.to_string(),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ));
        }
        lines.push(Line::from(spans));
    }

    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        TAGLINE,
        Style::default().fg(scale_color(Color::Rgb(0x80, 0x80, 0x80), 180)),
    )));

    let body_h = lines.len() as u16;
    let inner = Rect {
        x: area.x,
        y: area.y + pad_y,
        width: area.width,
        height: body_h.min(area.height.saturating_sub(pad_y)),
    };
    f.render_widget(Paragraph::new(lines).alignment(Alignment::Left), inner);
}

// ── Phase 4: Void swallows ───────────────────────────────────────────────────

fn render_swallow(f: &mut Frame, area: Rect, t: f32, accent: Color) {
    // Letters get pulled toward the D (center-right), disappearing from
    // the outside in. The reveal_cols shrinks from both ends toward D's position.
    let wordmark = WORDMARK;
    let wm_height = wordmark.len() as u16;
    let wm_width = wordmark.iter().map(|r| r.len()).max().unwrap_or(0) as u16;

    let pad_y = area.height.saturating_sub(wm_height + 2) / 2;
    let pad_x = area.width.saturating_sub(wm_width) / 2;

    // The void center (D position, roughly column 33).
    let void_center = 33usize;

    // t=0: full visible. t=1: everything gone.
    // Columns further from void_center disappear first.
    let max_dist = void_center.max(wm_width as usize - void_center) as f32;
    let kill_radius = ((1.0 - t) * max_dist) as usize;

    let intensity = ((1.0 - t) * 255.0) as u8;
    let _fade_color = scale_color(Color::Rgb(0xe5, 0xe5, 0xe5), intensity);
    let _v_fade = scale_color(accent, intensity);

    let v_range = 0..9usize;

    let mut lines: Vec<Line> = Vec::with_capacity(wm_height as usize + 2);
    lines.push(Line::raw(""));

    for row_str in wordmark {
        let mut spans: Vec<Span> = Vec::new();
        spans.push(Span::raw(" ".repeat(pad_x as usize)));

        let chars: Vec<char> = row_str.chars().collect();
        for (col, &ch) in chars.iter().enumerate() {
            let dist = if col > void_center {
                col - void_center
            } else {
                void_center - col
            };

            if dist > kill_radius {
                spans.push(Span::raw(" "));
                continue;
            }

            // Closer to void = stays longer but dims.
            let proximity = 1.0 - (dist as f32 / max_dist);
            let local_alpha = ((proximity * 0.5 + 0.5) * intensity as f32) as u8;
            let color = if v_range.contains(&col) {
                scale_color(accent, local_alpha)
            } else {
                scale_color(Color::Rgb(0xe5, 0xe5, 0xe5), local_alpha)
            };

            spans.push(Span::styled(
                ch.to_string(),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ));
        }
        lines.push(Line::from(spans));
    }

    // Tagline fading out.
    if intensity > 30 {
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled(
            TAGLINE,
            Style::default().fg(scale_color(Color::Rgb(0x80, 0x80, 0x80), intensity / 2)),
        )));
    }

    let body_h = lines.len() as u16;
    let inner = Rect {
        x: area.x,
        y: area.y + pad_y,
        width: area.width,
        height: body_h.min(area.height.saturating_sub(pad_y)),
    };
    f.render_widget(Paragraph::new(lines).alignment(Alignment::Left), inner);
}

// ── Phase 5: Collapse to nothing ─────────────────────────────────────────────

fn render_collapse(f: &mut Frame, area: Rect, t: f32) {
    // A single point of light shrinks and disappears.
    if t > 0.7 {
        return; // Pure black — UI will appear next frame.
    }

    let intensity = ((1.0 - t / 0.7) * 200.0) as u8;
    let mid_y = area.y + area.height / 2;
    let mid_x = area.x + area.width / 2;

    let dot_width = ((1.0 - t / 0.7) * 3.0) as u16;
    let dot_char = if t < 0.3 { '•' } else { '·' };
    let dot: String = std::iter::repeat(dot_char).take(dot_width.max(1) as usize).collect();

    let color = Color::Rgb(intensity, intensity, intensity);
    let line = Line::from(Span::styled(dot, Style::default().fg(color)));
    let rect = Rect {
        x: mid_x.saturating_sub(dot_width / 2),
        y: mid_y,
        width: dot_width.max(1),
        height: 1,
    };
    f.render_widget(Paragraph::new(vec![line]).alignment(Alignment::Center), rect);
}

// ── Utility ──────────────────────────────────────────────────────────────────

fn scale_color(c: Color, alpha: u8) -> Color {
    let scale = |b: u8| ((b as u16 * alpha as u16) / 255) as u8;
    match c {
        Color::Rgb(r, g, b) => Color::Rgb(scale(r), scale(g), scale(b)),
        other => other,
    }
}
