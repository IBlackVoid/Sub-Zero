use super::parse_srt_timing_line;
use super::time::{interval_overlap_seconds, Interval};
use crate::engine::srt::parse_srt_file;
use std::path::Path;

pub(super) fn verify_srt_against_audio(
    output_srt: &Path,
    audio_wav: &Path,
    threshold_db: f64,
    min_silence: f64,
    pad: f64,
    min_speech_overlap: f64,
) -> Result<String, String> {
    let cues = parse_srt_file(output_srt).map_err(|error| error.to_string())?;
    let mut max_end = 0.0f64;
    for cue in &cues {
        let (_, end) = parse_srt_timing_line(&cue.timing)?;
        if end > max_end {
            max_end = end;
        }
    }
    let analysis_end = (max_end + 1.0).max(0.0);
    let speech = crate::engine::transcribe::detect_speech_intervals_from_wav(
        audio_wav,
        threshold_db,
        min_silence,
        pad,
        Some(analysis_end),
    )?;

    let mut issues = Vec::<String>::new();
    let mut last_end = 0.0f64;

    for cue in &cues {
        let (start, end) = parse_srt_timing_line(&cue.timing)?;
        if start < last_end {
            issues.push(format!(
                "non-monotonic cue {}: {} (prev_end={:.3})",
                cue.index, cue.timing, last_end
            ));
        }
        last_end = end;

        let dur = (end - start).max(0.000_001);
        let cue_interval = Interval { start, end };
        let mut overlap = 0.0f64;
        for s in &speech {
            overlap += interval_overlap_seconds(
                cue_interval,
                Interval {
                    start: s.start,
                    end: s.end,
                },
            );
        }
        let ratio = overlap / dur;
        if ratio < min_speech_overlap {
            issues.push(format!(
                "low speech overlap cue {}: {:.2}% ({}): {}",
                cue.index,
                ratio * 100.0,
                cue.timing,
                cue.text.replace('\n', " / ")
            ));
        }
    }

    let mut report = String::new();
    report.push_str("sub-zero verify report\n");
    report.push_str(&format!("srt: {}\n", output_srt.display()));
    report.push_str(&format!("audio: {}\n", audio_wav.display()));
    report.push_str(&format!(
        "vad: threshold_db={threshold_db} min_silence={min_silence} pad={pad}\n"
    ));
    report.push_str(&format!("min_speech_overlap: {min_speech_overlap}\n\n"));

    if issues.is_empty() {
        report.push_str("status: ok\n");
    } else {
        report.push_str(&format!("status: issues={}\n", issues.len()));
        for issue in issues {
            report.push_str("- ");
            report.push_str(&issue);
            report.push('\n');
        }
    }

    Ok(report)
}
