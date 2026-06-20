//! Inter-cue ASR decode-loop collapse.
//!
//! whisper.cpp (and faster-whisper) decoders occasionally enter a
//! *decode loop* over a non-speech / music stretch: the model emits the
//! same short phrase for every window, producing a run of many identical
//! consecutive cues with a uniform cadence (e.g. 56 cues of
//! `いや、調べてもいいの?`, each exactly 2.000 s, spanning two minutes).
//!
//! Such a run is low-entropy and looks speech-like, so it slips past the
//! per-segment `--no-speech-thold` / `--entropy-thold` guards inside the
//! transcriber. Left in place it poisons translation: a faithful MT pass
//! reproduces the loop, the quality assessor scores it ~0.0 for lexical
//! collapse, and the whole job fails the MT quality floor — turning a
//! good 90-minute transcript into zero output over a ~2-minute artifact.
//!
//! The collapse here is the *upstream* fix: detect the unambiguous loop
//! signature on the source-language cues (before translation) and fold
//! each run down to a single cue spanning the run's full time range. The
//! engine still fails closed on genuinely degenerate output; this only
//! removes the narrow, mechanical artifact that should never have been
//! emitted.
//!
//! This guard is intra-modal and language-agnostic: it keys on the
//! *normalized cue text* (trim + lowercase + whitespace-collapsed) being
//! byte-identical across consecutive cues, so it works for Japanese,
//! English, or any source language without per-language tuning.

use super::confidence::CueAsrConfidence;
use super::time::{format_srt_timing_line, parse_srt_timing_line};
use crate::engine::srt::SubtitleCue;

/// Minimum number of *consecutive, identically-normalized* cues that
/// constitutes a decode loop.
///
/// Genuine dialogue effectively never repeats the same line four times
/// back to back: an intentional `Run!` / `Run!` is a 2x repeat, and even
/// an emphatic triple ("No. No. No.") tops out at 3. Real whisper decode
/// loops, by contrast, run for *tens* of cues (the observed cases were
/// 56x and 34x). A threshold of 4 sits comfortably above any plausible
/// natural repetition while still well below the artifact, giving a wide
/// safety margin against false positives. We deliberately do *not* go
/// lower (2 or 3) precisely because those lengths can occur in real
/// speech.
pub(super) const ASR_LOOP_MIN_RUN: usize = 4;

/// Outcome of an [`collapse_asr_decode_loops`] pass.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct AsrLoopCollapseStats {
    /// Number of distinct decode-loop runs that were collapsed.
    pub(super) runs: usize,
    /// Total cues removed (each run of length `n` removes `n - 1`).
    pub(super) cues_removed: usize,
}

impl AsrLoopCollapseStats {
    pub(super) fn collapsed_anything(&self) -> bool {
        self.runs > 0
    }
}

/// Normalize cue text for loop detection: trim, collapse internal
/// whitespace to single spaces, and lowercase (ASCII-fold; non-ASCII is
/// left as-is, which is correct for CJK where case does not apply).
fn normalized_key(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Collapse runs of `ASR_LOOP_MIN_RUN`-or-more consecutive cues whose
/// normalized text is identical into a single cue spanning the run's full
/// time range (`first.start --> last.end`), keeping one copy of the text.
///
/// `confidence`, when present, is kept index-aligned with `cues` in
/// lockstep: a collapsed run keeps the first cue's confidence entry
/// (`CueAsrConfidence` is `Copy`).
///
/// Runs shorter than the threshold are left completely untouched, so a
/// deliberate 2x or 3x repeat survives verbatim.
pub(super) fn collapse_asr_decode_loops(
    cues: &mut Vec<SubtitleCue>,
    confidence: &mut Option<Vec<Option<CueAsrConfidence>>>,
) -> AsrLoopCollapseStats {
    let mut stats = AsrLoopCollapseStats::default();
    if cues.len() < ASR_LOOP_MIN_RUN {
        return stats;
    }

    // If a confidence vector is present but does not line up with the
    // cues, we cannot safely re-align it; drop it rather than risk
    // emitting mismatched confidences. (The caller treats `None` as
    // "no confidence data", which is a benign fallback.)
    let conf = match confidence.take() {
        Some(c) if c.len() == cues.len() => Some(c),
        _ => None,
    };

    let mut out_cues: Vec<SubtitleCue> = Vec::with_capacity(cues.len());
    let mut out_conf: Option<Vec<Option<CueAsrConfidence>>> =
        conf.as_ref().map(|c| Vec::with_capacity(c.len()));

    let mut i = 0usize;
    while i < cues.len() {
        // Extend the run while the next cue normalizes to the same key.
        let key = normalized_key(&cues[i].text);
        let mut j = i;
        // An empty key (blank cue) is never treated as a loop: blank
        // runs are not the decode-loop signature and collapsing them
        // could swallow legitimate timing gaps.
        if !key.is_empty() {
            while j + 1 < cues.len() && normalized_key(&cues[j + 1].text) == key {
                j += 1;
            }
        }

        let run_len = j - i + 1;
        if run_len >= ASR_LOOP_MIN_RUN {
            // Collapse [i..=j] into one cue spanning first.start ->
            // last.end. Reuse the canonical timing parse/format so the
            // emitted timestamp matches the rest of the pipeline exactly.
            let merged_timing = match (
                parse_srt_timing_line(&cues[i].timing),
                parse_srt_timing_line(&cues[j].timing),
            ) {
                (Ok((start, _)), Ok((_, end))) => format_srt_timing_line(start, end.max(start)),
                // If either endpoint fails to parse (malformed
                // timing), fall back to the first cue's timing
                // verbatim rather than fabricating a span.
                _ => cues[i].timing.clone(),
            };

            out_cues.push(SubtitleCue {
                index: cues[i].index,
                timing: merged_timing,
                text: cues[i].text.clone(),
            });
            if let (Some(src), Some(dst)) = (conf.as_ref(), out_conf.as_mut()) {
                dst.push(src[i]);
            }

            stats.runs += 1;
            stats.cues_removed += run_len - 1;
        } else {
            // Sub-threshold run: copy every cue through untouched.
            for k in i..=j {
                out_cues.push(cues[k].clone());
                if let (Some(src), Some(dst)) = (conf.as_ref(), out_conf.as_mut()) {
                    dst.push(src[k]);
                }
            }
        }

        i = j + 1;
    }

    *cues = out_cues;
    *confidence = out_conf.take();
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(index: usize, start_s: u64, end_s: u64, text: &str) -> SubtitleCue {
        SubtitleCue {
            index,
            timing: format!(
                "00:{:02}:{:02},000 --> 00:{:02}:{:02},000",
                start_s / 60,
                start_s % 60,
                end_s / 60,
                end_s % 60
            ),
            text: text.to_string(),
        }
    }

    fn conf(score: f64) -> Option<CueAsrConfidence> {
        Some(CueAsrConfidence {
            score,
            avg_logprob: -0.1,
            no_speech_prob: 0.0,
            compression_ratio: 1.0,
            word_prob_mean: 0.9,
            low_word_prob_ratio: 0.0,
            suspicious: false,
        })
    }

    #[test]
    fn collapses_run_of_six_to_one_with_merged_span() {
        let mut cues: Vec<SubtitleCue> = (0u64..6)
            .map(|n| cue((n as usize) + 1, n * 2, n * 2 + 2, "loop phrase"))
            .collect();
        let mut confidence = None;

        let stats = collapse_asr_decode_loops(&mut cues, &mut confidence);

        assert_eq!(stats.runs, 1);
        assert_eq!(stats.cues_removed, 5);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "loop phrase");
        // first.start (0s) -> last.end (12s).
        assert_eq!(cues[0].timing, "00:00:00,000 --> 00:00:12,000");
    }

    #[test]
    fn leaves_two_repeat_untouched() {
        let original = vec![
            cue(1, 0, 1, "Run!"),
            cue(2, 1, 2, "Run!"),
            cue(3, 2, 3, "Keep going"),
        ];
        let mut cues = original.clone();
        let mut confidence = None;

        let stats = collapse_asr_decode_loops(&mut cues, &mut confidence);

        assert!(!stats.collapsed_anything());
        assert_eq!(cues, original);
    }

    #[test]
    fn leaves_triple_repeat_untouched() {
        // A threshold of 4 must not fire on an emphatic triple.
        let original = vec![
            cue(1, 0, 1, "No."),
            cue(2, 1, 2, "No."),
            cue(3, 2, 3, "No."),
        ];
        let mut cues = original.clone();
        let mut confidence = None;

        collapse_asr_decode_loops(&mut cues, &mut confidence);

        assert_eq!(cues, original);
    }

    #[test]
    fn collapses_two_separate_loops_and_preserves_middle() {
        let mut cues = vec![
            // First loop: 5x.
            cue(1, 0, 2, "phrase a"),
            cue(2, 2, 4, "phrase a"),
            cue(3, 4, 6, "phrase a"),
            cue(4, 6, 8, "phrase a"),
            cue(5, 8, 10, "phrase a"),
            // Genuine dialogue in between (must survive).
            cue(6, 10, 12, "real line one"),
            cue(7, 12, 14, "real line two"),
            // Second loop: 4x.
            cue(8, 14, 16, "phrase b"),
            cue(9, 16, 18, "phrase b"),
            cue(10, 18, 20, "phrase b"),
            cue(11, 20, 22, "phrase b"),
        ];
        let mut confidence = None;

        let stats = collapse_asr_decode_loops(&mut cues, &mut confidence);

        assert_eq!(stats.runs, 2);
        assert_eq!(stats.cues_removed, 4 + 3);
        // 1 (loop a) + 2 (dialogue) + 1 (loop b) = 4 surviving cues.
        assert_eq!(cues.len(), 4);
        assert_eq!(cues[0].text, "phrase a");
        assert_eq!(cues[0].timing, "00:00:00,000 --> 00:00:10,000");
        assert_eq!(cues[1].text, "real line one");
        assert_eq!(cues[2].text, "real line two");
        assert_eq!(cues[3].text, "phrase b");
        assert_eq!(cues[3].timing, "00:00:14,000 --> 00:00:22,000");
    }

    #[test]
    fn confidence_vector_stays_aligned() {
        let mut cues = vec![
            cue(1, 0, 2, "loop"),
            cue(2, 2, 4, "loop"),
            cue(3, 4, 6, "loop"),
            cue(4, 6, 8, "loop"),
            cue(5, 8, 10, "after"),
        ];
        // Distinct scores so we can prove the surviving entry is the
        // run's *first* confidence.
        let mut confidence = Some(vec![
            conf(0.91),
            conf(0.40),
            conf(0.30),
            conf(0.20),
            conf(0.88),
        ]);

        let stats = collapse_asr_decode_loops(&mut cues, &mut confidence);

        assert_eq!(stats.runs, 1);
        assert_eq!(cues.len(), 2);
        let conf = confidence.expect("confidence should be retained");
        assert_eq!(conf.len(), cues.len());
        // Surviving collapsed cue keeps the first run entry's score.
        assert_eq!(conf[0].unwrap().score, 0.91);
        // The trailing non-loop cue's confidence is preserved.
        assert_eq!(conf[1].unwrap().score, 0.88);
    }

    #[test]
    fn normalization_treats_whitespace_and_case_as_equal() {
        let mut cues = vec![
            cue(1, 0, 2, "Hello  World"),
            cue(2, 2, 4, "hello world"),
            cue(3, 4, 6, "HELLO WORLD"),
            cue(4, 6, 8, " hello   world "),
        ];
        let mut confidence = None;

        let stats = collapse_asr_decode_loops(&mut cues, &mut confidence);

        assert_eq!(stats.runs, 1);
        assert_eq!(cues.len(), 1);
        // The surviving cue keeps the *original* (first) text, not the key.
        assert_eq!(cues[0].text, "Hello  World");
    }

    #[test]
    fn blank_cue_run_is_not_collapsed() {
        let original = vec![
            cue(1, 0, 1, ""),
            cue(2, 1, 2, ""),
            cue(3, 2, 3, ""),
            cue(4, 3, 4, ""),
            cue(5, 4, 5, ""),
        ];
        let mut cues = original.clone();
        let mut confidence = None;

        let stats = collapse_asr_decode_loops(&mut cues, &mut confidence);

        assert!(!stats.collapsed_anything());
        assert_eq!(cues, original);
    }
}
