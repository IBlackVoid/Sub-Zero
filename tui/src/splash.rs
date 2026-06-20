use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};
use std::time::{Duration, Instant};

// ── Duration ────────────────────────────────────────────────────────────────
/// Total splash: 6.5 s across six phases.
///
///   Phase 0  0.0–0.5 s   void breathes — darkness, faint particle drift
///   Phase 1  0.5–1.2 s   rift cracks open from center
///   Phase 2  1.2–3.0 s   letters emerge one-by-one (V → o → i → D → e → x)
///   Phase 3  3.0–4.2 s   hold — V shimmers, D breathes, particles orbit
///   Phase 4  4.2–5.7 s   void consumes (x first, D last)
///   Phase 5  5.7–6.5 s   collapse to singularity → black → UI
pub const SPLASH_DURATION: Duration = Duration::from_millis(6_500);

const P0: f32 = 0.077; // 0.5 / 6.5
const P1: f32 = 0.185; // 1.2 / 6.5
const P2: f32 = 0.462; // 3.0 / 6.5
const P3: f32 = 0.646; // 4.2 / 6.5
const P4: f32 = 0.877; // 5.7 / 6.5

// ── Brand ───────────────────────────────────────────────────────────────────
const BRAND_BLUE: Color = Color::Rgb(0x3b, 0x82, 0xf6);
const WHITE: Color = Color::Rgb(0xe5, 0xe5, 0xe5);

// ── Per-letter art ──────────────────────────────────────────────────────────
//
// 6 rows each, consistent display-width per letter.
// All characters are 1 terminal column wide (box-drawing + full-blocks).

const LV: [&str; 5] = [
    "▐█▌   ▐█▌", // 9
    " ▐█▌ ▐█▌ ",
    "  ▐███▌  ",
    "   ▐█▌   ",
    "    ▀    ",
];
const LO: [&str; 5] = [
    " ▄████▄ ", // 8
    "██    ██",
    "██    ██",
    "▐█▄  ▄█▌",
    " ▀████▀ ",
];
const LI: [&str; 5] = [
    "▐█▌", // 3
    "▐█▌",
    "▐█▌",
    "▐█▌",
    " ▀ ",
];
const LD: [&str; 5] = [
    "█████▄  ", // 8
    "██  ▐█▌ ",
    "██  ▐█▌ ",
    "██  ▄█▀ ",
    "█████▀  ",
];
const LE: [&str; 5] = [
    "███████▌", // 8
    "██▌     ",
    "█████▌  ",
    "██▌     ",
    "███████▌",
];
const LX: [&str; 5] = [
    "▐█▌  ▐█▌", // 8
    " ▐████▌ ",
    "  ▐██▌  ",
    " ▐████▌ ",
    "▐█▌  ▐█▌",
];

const LETTERS: [&[&str; 5]; 6] = [&LV, &LO, &LI, &LD, &LE, &LX];
const WIDTHS: [u16; 6] = [9, 8, 3, 8, 8, 8];
const GAP: u16 = 3;
const WM_H: u16 = 5;
/// 9+3 + 8+3 + 3+3 + 8+3 + 8+3 + 8 = 59
const WM_W: u16 = 59;

const WAVES: [&str; 5] = ["≈≈∿≋~", "∿≋~≈≈", "~∿≈≋∿", "≋~∿≈≋", "≈∿≋~≈"];
const PARTICLES: [char; 8] = ['·', '∙', '•', '◦', '◌', '∘', '⋅', '∶'];
const TAGLINE: &str = "the void indexes your voice";

// ── Public API ──────────────────────────────────────────────────────────────

pub fn render(f: &mut Frame, area: Rect, started_at: Instant, accent: Color) {
    let elapsed = started_at.elapsed();
    let t = (elapsed.as_secs_f32() / SPLASH_DURATION.as_secs_f32()).clamp(0.0, 1.0);
    let accent = guard_visibility(accent);

    // Background particle field (all phases except late collapse).
    if t < P4 + 0.04 {
        draw_particles(f, area, elapsed, accent, t);
    }

    match () {
        _ if t < P0 => phase_breathe(f, area, t / P0, accent),
        _ if t < P1 => phase_rift(f, area, (t - P0) / (P1 - P0), accent, elapsed),
        _ if t < P2 => phase_emerge(f, area, (t - P1) / (P2 - P1), accent, elapsed),
        _ if t < P3 => phase_hold(f, area, (t - P2) / (P3 - P2), accent, elapsed),
        _ if t < P4 => phase_consume(f, area, (t - P3) / (P4 - P3), accent, elapsed),
        _ => phase_collapse(f, area, (t - P4) / (1.0 - P4), accent),
    }
}

// ── Accent guard ────────────────────────────────────────────────────────────
/// Replace any accent that would be invisible on a dark background.
fn guard_visibility(c: Color) -> Color {
    match c {
        Color::Rgb(r, g, b) if (r as u16 + g as u16 + b as u16) < 180 => BRAND_BLUE,
        Color::Reset => BRAND_BLUE,
        other => other,
    }
}

// ── Layout ──────────────────────────────────────────────────────────────────

/// Content block: wave(1) + gap(1) + wordmark(5) + gap(1) + wave(1) + gap(1) + tagline(1) = 11 rows
const CONTENT_H: u16 = 11;

fn origin(area: Rect) -> (u16, u16) {
    let x = area.x + area.width.saturating_sub(WM_W) / 2;
    let y = area.y + area.height.saturating_sub(CONTENT_H) / 2;
    (x, y)
}

fn letter_x(base_x: u16, idx: usize) -> u16 {
    let mut x = base_x;
    for width in &WIDTHS[..idx] {
        x += width + GAP;
    }
    x
}

// ── Particles ───────────────────────────────────────────────────────────────

fn draw_particles(f: &mut Frame, area: Rect, elapsed: Duration, accent: Color, t: f32) {
    let ms = elapsed.as_millis() as u64;
    let count: usize = if t < P1 {
        35
    } else if t < P3 {
        55
    } else {
        80
    };

    let d_cx = (area.x + area.width / 2) as f32 + 6.0; // approx D center
    let d_cy = (area.y + area.height / 2) as f32;

    for i in 0..count {
        let seed = (i as u64).wrapping_mul(7919).wrapping_add(1);
        let mut px = ((seed.wrapping_mul(31).wrapping_add(ms / 220)) % area.width as u64) as f32;
        let mut py = ((seed.wrapping_mul(47).wrapping_add(ms / 310)) % area.height as u64) as f32;

        // During consume/collapse, particles gravitate toward D.
        if t > P3 {
            let pull = ((t - P3) / (1.0 - P3)).powi(2);
            px += (d_cx - area.x as f32 - px) * pull;
            py += (d_cy - area.y as f32 - py) * pull;
        }

        let ix = area.x + (px as u16).min(area.width.saturating_sub(1));
        let iy = area.y + (py as u16).min(area.height.saturating_sub(1));

        let twinkle = ((ms as f32 / 110.0 + i as f32 * 2.3).sin() * 0.5 + 0.5) * 55.0 + 8.0;
        let ch = PARTICLES[i % PARTICLES.len()];
        let color = sc(accent, twinkle as u8);

        put(f, ix, iy, &ch.to_string(), Style::default().fg(color));
    }
}

// ── Phase 0 — Void breathes ─────────────────────────────────────────────────

fn phase_breathe(f: &mut Frame, area: Rect, t: f32, accent: Color) {
    let cx = area.x + area.width / 2;
    let cy = area.y + area.height / 2;

    let pulse = (t * std::f32::consts::PI).sin();
    let a = (pulse * 120.0) as u8;
    let color = sc(accent, a);
    let s = Style::default().fg(color).add_modifier(Modifier::BOLD);

    let glyph = if t < 0.35 {
        "  ·  "
    } else if t < 0.65 {
        " ·•· "
    } else {
        "·•◦•·"
    };
    let w = glyph.chars().count() as u16;
    put(f, cx.saturating_sub(w / 2), cy, glyph, s);
}

// ── Phase 1 — Rift ──────────────────────────────────────────────────────────

fn phase_rift(f: &mut Frame, area: Rect, t: f32, accent: Color, elapsed: Duration) {
    let cx = area.x + area.width / 2;
    let cy = area.y + area.height / 2;
    let ms = elapsed.as_millis() as f32;

    let max_half = (WM_W / 2 + 5) as f32;
    let half = (ease_out(t) * max_half) as u16;
    let a = (ease_out(t) * 255.0) as u8;

    // Main rift — character evolves with intensity.
    let ch = if t < 0.2 {
        '·'
    } else if t < 0.45 {
        '─'
    } else if t < 0.7 {
        '━'
    } else {
        '═'
    };
    let rift: String = ch.to_string().repeat((half * 2) as usize);
    let x = cx.saturating_sub(half);
    let color = sc(accent, a);
    put(
        f,
        x,
        cy,
        &rift,
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    );

    // Parallel glow lines above & below.
    if t > 0.35 {
        let g = ((t - 0.35) / 0.65 * 70.0) as u8;
        let gh = (half as f32 * 0.65) as u16;
        let gx = cx.saturating_sub(gh);
        let grift: String = "─".repeat((gh * 2) as usize);
        let gs = Style::default().fg(sc(accent, g));
        for dy in [cy.wrapping_sub(1), cy + 1] {
            if dy > area.y && dy < area.y + area.height {
                put(f, gx, dy, &grift, gs);
            }
        }
    }

    // Flickering bright spots along the rift.
    if t > 0.5 {
        let n = ((t - 0.5) * 16.0) as usize;
        for i in 0..n {
            let fx = x + ((i as u64 * 7919 + ms as u64 / 70) % (half as u64 * 2).max(1)) as u16;
            if fx < area.x + area.width {
                put(
                    f,
                    fx,
                    cy,
                    "█",
                    Style::default()
                        .fg(sc(accent, 220))
                        .add_modifier(Modifier::BOLD),
                );
            }
        }
    }

    // Outer sparks.
    if t > 0.6 {
        let spark_a = ((t - 0.6) / 0.4 * 100.0) as u8;
        for (i, &dy) in [cy.wrapping_sub(2), cy + 2].iter().enumerate() {
            if dy > area.y && dy < area.y + area.height {
                let sx = cx.saturating_sub(3) + ((ms as u64 / 90 + i as u64 * 13) % 7) as u16;
                if sx < area.x + area.width {
                    put(f, sx, dy, "∙", Style::default().fg(sc(accent, spark_a)));
                }
            }
        }
    }
}

// ── Phase 2 — Letters emerge one by one ─────────────────────────────────────

fn phase_emerge(f: &mut Frame, area: Rect, t: f32, accent: Color, elapsed: Duration) {
    let (ox, oy) = origin(area);
    let ms = elapsed.as_millis() as f32;
    let wm_y = oy + 2; // skip top waveform + gap

    // Per-letter appear timing: V first, x last.
    let appear: [f32; 6] = [0.00, 0.13, 0.24, 0.40, 0.60, 0.76];
    let dur = 0.22_f32;

    // ── Top waveform (tied to V) ────────────────────────────────────────────
    let va = fade_in(t, appear[0], dur);
    if va > 0.1 {
        let wi = (elapsed.as_millis() / 100) as usize;
        let w = WAVES[wi % WAVES.len()];
        let s = Style::default().fg(sc(accent, (va * 220.0) as u8));
        put(f, ox, oy, w, s);
    }

    // ── Wordmark rows ───────────────────────────────────────────────────────
    for row in 0..WM_H as usize {
        let base_y = wm_y + row as u16;
        if base_y >= area.y + area.height {
            break;
        }

        let mut lx = ox;
        for (li, &letter) in LETTERS.iter().enumerate() {
            let alpha = fade_in(t, appear[li], dur);
            let w = WIDTHS[li];

            if alpha <= 0.0 {
                lx += w + GAP;
                continue;
            }

            // Vertical slide: emerge from 2 rows below.
            let slide = ((1.0 - alpha) * 2.0) as u16;
            let ry = base_y + slide;
            if ry >= area.y + area.height {
                lx += w + GAP;
                continue;
            }

            render_letter_row(f, li, letter[row], lx, ry, alpha, accent, ms, area);
            lx += w + GAP;
        }
    }

    // ── Bottom waveform ─────────────────────────────────────────────────────
    if va > 0.35 {
        let wi = (elapsed.as_millis() / 120) as usize;
        let w = WAVES[(wi + 2) % WAVES.len()];
        let wy = wm_y + WM_H + 1;
        if wy < area.y + area.height {
            let s = Style::default().fg(sc(accent, ((va - 0.35) / 0.65 * 150.0) as u8));
            put(f, ox, wy, w, s);
        }
    }

    // ── Tagline ─────────────────────────────────────────────────────────────
    let tag_a = ((t - 0.88) / 0.12).clamp(0.0, 1.0);
    if tag_a > 0.0 {
        let tag_y = wm_y + WM_H + 3;
        if tag_y < area.y + area.height {
            let tw = TAGLINE.chars().count() as u16;
            let tx = area.x + area.width.saturating_sub(tw) / 2;
            let s = Style::default().fg(sc(Color::Rgb(0x58, 0x58, 0x58), (tag_a * 220.0) as u8));
            put(f, tx, tag_y, TAGLINE, s);
        }
    }
}

// ── Phase 3 — Hold + breathe ────────────────────────────────────────────────

fn phase_hold(f: &mut Frame, area: Rect, t: f32, accent: Color, elapsed: Duration) {
    let (ox, oy) = origin(area);
    let ms = elapsed.as_millis() as f32;
    let wm_y = oy + 2;

    // Top waveform — V's voice.
    let wi = (elapsed.as_millis() / 80) as usize;
    let w = WAVES[wi % WAVES.len()];
    put(f, ox, oy, w, Style::default().fg(accent));

    // Full wordmark, fully visible. V shimmers, D breathes.
    for row in 0..WM_H as usize {
        let y = wm_y + row as u16;
        if y >= area.y + area.height {
            break;
        }

        let mut lx = ox;
        for (li, &letter) in LETTERS.iter().enumerate() {
            render_letter_row(f, li, letter[row], lx, y, 1.0, accent, ms, area);
            lx += WIDTHS[li] + GAP;
        }
    }

    // Bottom waveform.
    let w2 = WAVES[(wi + 3) % WAVES.len()];
    let wy = wm_y + WM_H + 1;
    if wy < area.y + area.height {
        put(f, ox, wy, w2, Style::default().fg(sc(accent, 170)));
    }

    // Tagline.
    let tag_y = wm_y + WM_H + 3;
    if tag_y < area.y + area.height {
        let tw = TAGLINE.chars().count() as u16;
        let tx = area.x + area.width.saturating_sub(tw) / 2;
        put(
            f,
            tx,
            tag_y,
            TAGLINE,
            Style::default().fg(Color::Rgb(0x58, 0x58, 0x58)),
        );
    }

    // D void vortex — tiny particles orbiting inside D's hollow.
    let dx = letter_x(ox, 3);
    let d_cx = dx + 4; // center col of D
    let d_cy = wm_y + 2; // center row of D (5-row wordmark)
    for i in 0..3u32 {
        let angle = ms / 250.0 + (i as f32 * std::f32::consts::TAU / 3.0);
        let px = d_cx as f32 + angle.cos() * 1.5;
        let py = d_cy as f32 + (angle.sin() * 0.6);
        let ix = px as u16;
        let iy = py as u16;
        if ix >= area.x && ix < area.x + area.width && iy >= area.y && iy < area.y + area.height {
            let p = PARTICLES[(i as usize + wi) % PARTICLES.len()];
            put(
                f,
                ix,
                iy,
                &p.to_string(),
                Style::default().fg(sc(accent, 90)),
            );
        }
    }

    // Subtle horizontal tendrils extending from the wordmark edges.
    let tendril_len = ((t.sin() * 3.0).abs() + 2.0) as u16;
    let mid_y = wm_y + WM_H / 2;
    if mid_y < area.y + area.height {
        // Left tendril (from V).
        let left: String = "─".repeat(tendril_len as usize);
        let lx_start = ox.saturating_sub(tendril_len + 1);
        if lx_start >= area.x {
            put(
                f,
                lx_start,
                mid_y,
                &left,
                Style::default().fg(sc(accent, 35)),
            );
        }
        // Right tendril (from X).
        let rx_start = ox + WM_W + 1;
        if rx_start + tendril_len < area.x + area.width {
            put(
                f,
                rx_start,
                mid_y,
                &left,
                Style::default().fg(sc(accent, 35)),
            );
        }
    }
}

// ── Phase 4 — Void consumes ─────────────────────────────────────────────────

fn phase_consume(f: &mut Frame, area: Rect, t: f32, accent: Color, elapsed: Duration) {
    let (ox, oy) = origin(area);
    let ms = elapsed.as_millis() as f32;
    let wm_y = oy + 2;

    // Per-letter death schedule: x → e → i → o → V → D.
    let death: [f32; 6] = [
        0.52, // V — penultimate
        0.40, // o
        0.26, // i
        0.72, // D — the void dies last
        0.14, // e
        0.00, // x — first consumed
    ];
    let death_dur = 0.24_f32;

    for row in 0..WM_H as usize {
        let y = wm_y + row as u16;
        if y >= area.y + area.height {
            break;
        }

        let mut lx = ox;
        for (li, &letter) in LETTERS.iter().enumerate() {
            let alive = die_out(t, death[li], death_dur);

            if alive <= 0.01 {
                lx += WIDTHS[li] + GAP;
                continue;
            }

            let a = (alive * 255.0) as u8;
            let chars: Vec<char> = letter[row].chars().collect();

            for (col, &ch) in chars.iter().enumerate() {
                let cx = lx + col as u16;
                if cx >= area.x + area.width {
                    break;
                }
                if ch == ' ' {
                    continue;
                }

                // Below 30% life: characters fragment into drifting particles.
                if alive < 0.30 && (col + row) % 3 == 0 {
                    let p = PARTICLES[(col + row + (ms as usize / 100)) % PARTICLES.len()];
                    let pc = sc(accent, (alive * 140.0) as u8);
                    put(f, cx, y, &p.to_string(), Style::default().fg(pc));
                    continue;
                }

                // Jitter near death.
                let jitter = if alive < 0.45 {
                    (ms / 35.0 + col as f32 * 3.7 + row as f32).sin().abs() * 0.5 + 0.5
                } else {
                    1.0
                };

                let color = ltr_color(li, col, (a as f32 * jitter) as u8, accent, ms);
                put(
                    f,
                    cx,
                    y,
                    &ch.to_string(),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                );
            }

            lx += WIDTHS[li] + GAP;
        }
    }

    // Waveforms fade fast.
    let wfade = (1.0 - t * 1.8).max(0.0);
    if wfade > 0.0 {
        let wi = (elapsed.as_millis() / 100) as usize;
        let w = WAVES[wi % WAVES.len()];
        put(
            f,
            ox,
            oy,
            w,
            Style::default().fg(sc(accent, (wfade * 200.0) as u8)),
        );
    }

    // Tagline fades faster.
    let tfade = (1.0 - t * 2.5).max(0.0);
    if tfade > 0.0 {
        let tag_y = wm_y + WM_H + 3;
        if tag_y < area.y + area.height {
            let tw = TAGLINE.chars().count() as u16;
            let tx = area.x + area.width.saturating_sub(tw) / 2;
            put(
                f,
                tx,
                tag_y,
                TAGLINE,
                Style::default().fg(sc(Color::Rgb(0x58, 0x58, 0x58), (tfade * 200.0) as u8)),
            );
        }
    }
}

// ── Phase 5 — Collapse ──────────────────────────────────────────────────────

fn phase_collapse(f: &mut Frame, area: Rect, t: f32, accent: Color) {
    if t > 0.85 {
        return;
    } // pure black → UI appears

    let cx = area.x + area.width / 2;
    let cy = area.y + area.height / 2;
    let fade = 1.0 - t / 0.85;
    let a = (fade * 240.0) as u8;
    let radius = (fade * 5.0) as u16;
    let color = sc(accent, a);
    let s = Style::default().fg(color).add_modifier(Modifier::BOLD);

    if radius == 0 {
        put(f, cx, cy, "•", s);
    } else {
        let ring: String = format!(
            "{}•{}",
            "·".repeat(radius as usize),
            "·".repeat(radius as usize),
        );
        let w = ring.chars().count() as u16;
        put(f, cx.saturating_sub(w / 2), cy, &ring, s);
    }
}

// ── Letter rendering ────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)] // draw helper: positional render params are clearer than a one-off struct
fn render_letter_row(
    f: &mut Frame,
    li: usize, // letter index (0=V .. 5=X)
    row_str: &str,
    x: u16,
    y: u16,
    alpha: f32,
    accent: Color,
    ms: f32,
    area: Rect,
) {
    let a = (alpha * 255.0) as u8;
    let mut spans: Vec<Span<'static>> = Vec::new();

    for (col, ch) in row_str.chars().enumerate() {
        if ch == ' ' {
            spans.push(Span::raw(" "));
        } else {
            let color = ltr_color(li, col, a, accent, ms);
            spans.push(Span::styled(
                ch.to_string(),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ));
        }
    }

    let w = row_str.chars().count() as u16;
    let clipped_w = w.min(area.x + area.width - x.min(area.x + area.width));
    if clipped_w == 0 {
        return;
    }

    let rect = Rect {
        x,
        y,
        width: clipped_w,
        height: 1,
    };
    f.render_widget(Paragraph::new(Line::from(spans)), rect);
}

/// Per-letter color logic.
/// V = accent with sine shimmer.
/// D = white with slow breathing pulse.
/// Others = clean white.
fn ltr_color(li: usize, col: usize, alpha: u8, accent: Color, ms: f32) -> Color {
    match li {
        0 => {
            // V: accent, shimmering sine wave across columns.
            let shimmer = (ms / 90.0 + col as f32 * 0.7).sin() * 0.22 + 0.78;
            sc(accent, (shimmer * alpha as f32) as u8)
        }
        3 => {
            // D: white with a slow breathing luminance oscillation.
            let breathe = (ms / 450.0).sin() * 0.10 + 0.90;
            sc(WHITE, (breathe * alpha as f32) as u8)
        }
        _ => sc(WHITE, alpha),
    }
}

// ── Easing ──────────────────────────────────────────────────────────────────

fn ease_out(t: f32) -> f32 {
    1.0 - (1.0 - t).powi(3)
}
fn ease_in(t: f32) -> f32 {
    t * t * t
}

fn fade_in(t: f32, start: f32, dur: f32) -> f32 {
    if t < start {
        return 0.0;
    }
    ease_out(((t - start) / dur).min(1.0))
}

fn die_out(t: f32, start: f32, dur: f32) -> f32 {
    if t < start {
        return 1.0;
    }
    1.0 - ease_in(((t - start) / dur).min(1.0))
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Scale an RGB colour by `a/255`. Non-RGB passes through unchanged.
fn sc(c: Color, a: u8) -> Color {
    match c {
        Color::Rgb(r, g, b) => Color::Rgb(
            ((r as u16 * a as u16) / 255) as u8,
            ((g as u16 * a as u16) / 255) as u8,
            ((b as u16 * a as u16) / 255) as u8,
        ),
        other => other,
    }
}

/// Paint a string at an exact (x, y) with the given style. One line, no wrap.
fn put(f: &mut Frame, x: u16, y: u16, text: &str, style: Style) {
    let w = text.chars().count() as u16;
    if w == 0 {
        return;
    }
    let rect = Rect {
        x,
        y,
        width: w,
        height: 1,
    };
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(text.to_string(), style))),
        rect,
    );
}
