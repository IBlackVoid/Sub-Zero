//! Easter-egg slot directory layout — kept in its own module so the
//! two filesystem roots holding encrypted assets are referenced by one
//! canonical pair of constants instead of inlined string literals
//! throughout `app.rs` and `easter_egg.rs`. Looks small, but a search
//! for "assets/secret" on a future renamed layout returns here.
//!
//! Not dead code: see `app::App::handle_phrase` for both call sites.

/// First-slot asset directory. Maps to the public panel under the
/// primary unlock phrase. Layout is opaque to the rest of the program —
/// only `easter_egg::try_unlock` is allowed to interpret its contents.
pub const DIR_1: &str = "assets/secret/a";

/// Second-slot asset directory. Holds the solo-mode payload reached
/// via the secondary phrase. Same opacity contract as `DIR_1`.
pub const DIR_2: &str = "assets/secret/b";
