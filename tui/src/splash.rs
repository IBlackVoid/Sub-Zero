use ratatui::{
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};
use std::time::{Duration, Instant};

pub const SPLASH_DURATION: Duration = Duration::from_millis(1_400);

const LOGO: &[&str] = &[
    "  ███████ ██    ██ ██████          ███████ ███████ ██████   ██████  ",
    "  ██      ██    ██ ██   ██            ███  ██      ██   ██ ██    ██ ",
    "  ███████ ██    ██ ██████  ██████   ███    █████   ██████  ██    ██ ",
    "       ██ ██    ██ ██   ██         ███     ██      ██   ██ ██    ██ ",
    "  ███████  ██████  ██████          ███████ ███████ ██   ██  ██████  ",
];
const TAGLINE: &str = "offline-first subtitle translator · F.2 information-bottleneck core";

pub fn render(f: &mut Frame, area: Rect, started_at: Instant, accent: Color) {
    let elapsed = started_at.elapsed();
    let frac = (elapsed.as_secs_f32() / SPLASH_DURATION.as_secs_f32())
        .clamp(0.0, 1.0);
    let intensity = 1.0 - (1.0 - frac).powi(2);
    let alpha = (intensity * 255.0) as u8;
    let logo_color = scale_color(accent, alpha);
    let tagline_color = scale_color(Color::Rgb(0x80, 0x80, 0x80), alpha);

    let mut lines: Vec<Line> = Vec::with_capacity(LOGO.len() + 3);
    lines.push(Line::raw(""));
    for row in LOGO {
        lines.push(Line::from(Span::styled(
            *row,
            Style::default().fg(logo_color).add_modifier(Modifier::BOLD),
        )));
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        TAGLINE,
        Style::default().fg(tagline_color),
    )));

    let body_h = lines.len() as u16;
    let pad = area.height.saturating_sub(body_h) / 2;
    let inner = Rect {
        x: area.x,
        y: area.y + pad,
        width: area.width,
        height: body_h.min(area.height),
    };
    let p = Paragraph::new(lines).alignment(Alignment::Center);
    f.render_widget(p, inner);
}

fn scale_color(c: Color, alpha: u8) -> Color {
    let scale = |b: u8| ((b as u16 * alpha as u16) / 255) as u8;
    match c {
        Color::Rgb(r, g, b) => Color::Rgb(scale(r), scale(g), scale(b)),
        other => other,
    }
}
