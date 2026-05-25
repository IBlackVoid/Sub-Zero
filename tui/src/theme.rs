use ratatui::style::Color;

pub mod palette {
    use ratatui::style::Color;

    pub const ACCENT: Color = Color::Rgb(0xd0, 0x68, 0x20);
    pub const ACCENT_2: Color = Color::Rgb(0xe0, 0x98, 0x20);
    pub const PINK: Color = Color::Rgb(0xc0, 0x30, 0x70);
    pub const GREEN: Color = Color::Rgb(0x50, 0xc8, 0x70);
    pub const RED: Color = Color::Rgb(0xc0, 0x50, 0x50);
    pub const TEXT: Color = Color::Rgb(0xd0, 0xd0, 0xd0);
    pub const MUTED: Color = Color::Rgb(0x60, 0x60, 0x60);
    pub const FAINT: Color = Color::Rgb(0x32, 0x32, 0x32);
}

/// Brighten a colour toward white by ~30%. Used for the theme-switch
/// flash; returns the original colour for non-RGB variants.
pub fn brighten(c: Color) -> Color {
    match c {
        Color::Rgb(r, g, b) => Color::Rgb(
            r.saturating_add((255 - r) / 3),
            g.saturating_add((255 - g) / 3),
            b.saturating_add((255 - b) / 3),
        ),
        other => other,
    }
}

/// Linear-interpolate two RGB colours by `t` in [0, 1]. Non-RGB inputs
/// fall back to `a` because indexed/named colours are not mixable here.
pub fn lerp_color(a: Color, b: Color, t: f32) -> Color {
    let mix =
        |x: u8, y: u8| -> u8 { (x as f32 + (y as f32 - x as f32) * t).clamp(0.0, 255.0) as u8 };
    match (a, b) {
        (Color::Rgb(ar, ag, ab), Color::Rgb(br, bg, bb)) => {
            Color::Rgb(mix(ar, br), mix(ag, bg), mix(ab, bb))
        }
        (a, _) => a,
    }
}

/// Theme-driven accent colour for animation borders and title chrome.
/// Honours `NO_COLOR`: when the user has opted into a colourless terminal,
/// `Color::Reset` is returned so the host terminal's default chrome shows
/// through instead of an RGB accent.
pub fn theme_accent(theme: crate::easter_egg::Theme) -> Color {
    if crate::accessibility::Accessibility::from_env().no_color {
        return Color::Reset;
    }
    use crate::easter_egg::Theme as T;

    match theme {
        T::Default => palette::FAINT,
        T::MrRobot => Color::Rgb(0xff, 0x14, 0x14),
        T::SynthwavePurple => Color::Rgb(0xff, 0x4d, 0xff),
        T::AmberCrt => Color::Rgb(0xff, 0xb0, 0x00),
        T::KeimaBlue => Color::Rgb(0x59, 0xa6, 0xff),
    }
}
