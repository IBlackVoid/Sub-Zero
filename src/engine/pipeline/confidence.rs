use super::parse_srt_timing_line;
use super::time::{interval_overlap_seconds, Interval};
use crate::engine::srt::SubtitleCue;
use crate::engine::transcribe::QualityProfile;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub(super) struct CueAsrConfidence {
    pub(super) score: f64,
    pub(super) avg_logprob: f64,
    pub(super) no_speech_prob: f64,
    pub(super) compression_ratio: f64,
    pub(super) word_prob_mean: f64,
    pub(super) low_word_prob_ratio: f64,
    pub(super) suspicious: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LowConfidenceCueSpan {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) score: f64,
    pub(super) floor: f64,
}

#[derive(Debug, Clone, Copy)]
struct WhisperConfidenceSegment {
    start: f64,
    end: f64,
    avg_logprob: f64,
    no_speech_prob: f64,
    compression_ratio: f64,
    word_prob_mean: f64,
    low_word_prob_ratio: f64,
}

pub(super) fn load_cue_asr_confidence_from_whisper_json(
    srt_path: &Path,
    cues: &[SubtitleCue],
) -> Option<Vec<Option<CueAsrConfidence>>> {
    if cues.is_empty() {
        return Some(Vec::new());
    }

    let json_path = srt_path.with_extension("json");
    let raw = std::fs::read_to_string(&json_path).ok()?;
    let parsed: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let segments = parse_whisper_confidence_segments(&parsed)?;
    if segments.is_empty() {
        return None;
    }

    let mut mapped = Vec::<Option<CueAsrConfidence>>::with_capacity(cues.len());
    for cue in cues {
        let Ok((cue_start, cue_end)) = parse_srt_timing_line(&cue.timing) else {
            mapped.push(None);
            continue;
        };
        let cue_interval = Interval {
            start: cue_start,
            end: cue_end,
        };
        let mut weighted_score = 0.0f64;
        let mut weighted_logprob = 0.0f64;
        let mut weighted_no_speech = 0.0f64;
        let mut weighted_compression = 0.0f64;
        let mut weighted_word_prob = 0.0f64;
        let mut weighted_low_word_prob = 0.0f64;
        let mut total_overlap = 0.0f64;
        let mut suspicious = false;

        for segment in &segments {
            let overlap = interval_overlap_seconds(
                cue_interval,
                Interval {
                    start: segment.start,
                    end: segment.end,
                },
            );
            if overlap <= 0.0 {
                continue;
            }

            let score = segment_confidence_score(segment);
            weighted_score += score * overlap;
            weighted_logprob += segment.avg_logprob * overlap;
            weighted_no_speech += segment.no_speech_prob * overlap;
            weighted_compression += segment.compression_ratio * overlap;
            weighted_word_prob += segment.word_prob_mean * overlap;
            weighted_low_word_prob += segment.low_word_prob_ratio * overlap;
            total_overlap += overlap;
            suspicious |= segment_is_suspicious(segment);
        }

        if total_overlap <= 0.0 {
            mapped.push(None);
            continue;
        }

        mapped.push(Some(CueAsrConfidence {
            score: (weighted_score / total_overlap).clamp(0.0, 1.0),
            avg_logprob: weighted_logprob / total_overlap,
            no_speech_prob: weighted_no_speech / total_overlap,
            compression_ratio: weighted_compression / total_overlap,
            word_prob_mean: weighted_word_prob / total_overlap,
            low_word_prob_ratio: weighted_low_word_prob / total_overlap,
            suspicious,
        }));
    }

    let covered = mapped.iter().filter(|entry| entry.is_some()).count();
    if covered == 0 {
        return None;
    }

    eprintln!(
        "ibvoid-doom-qlock: loaded ASR confidence from {} (coverage={}/{})",
        json_path.display(),
        covered,
        cues.len()
    );
    Some(mapped)
}

fn parse_whisper_confidence_segments(
    root: &serde_json::Value,
) -> Option<Vec<WhisperConfidenceSegment>> {
    let segments = root.get("segments")?.as_array()?;
    let mut out = Vec::<WhisperConfidenceSegment>::with_capacity(segments.len());
    for segment in segments {
        let Some(start) = segment.get("start").and_then(serde_json::Value::as_f64) else {
            continue;
        };
        let Some(end) = segment.get("end").and_then(serde_json::Value::as_f64) else {
            continue;
        };
        if end <= start {
            continue;
        }
        let avg_logprob = segment
            .get("avg_logprob")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(-1.5);
        let no_speech_prob = segment
            .get("no_speech_prob")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let compression_ratio = segment
            .get("compression_ratio")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1.0)
            .max(0.0);
        let (word_prob_mean, low_word_prob_ratio) = parse_word_probability_stats(segment);
        out.push(WhisperConfidenceSegment {
            start,
            end,
            avg_logprob,
            no_speech_prob,
            compression_ratio,
            word_prob_mean,
            low_word_prob_ratio,
        });
    }
    Some(out)
}

fn parse_word_probability_stats(segment: &serde_json::Value) -> (f64, f64) {
    let Some(words) = segment.get("words").and_then(serde_json::Value::as_array) else {
        return (0.0, 0.0);
    };
    let mut sum = 0.0f64;
    let mut total = 0usize;
    let mut low = 0usize;
    for word in words {
        let Some(prob) = word.get("probability").and_then(serde_json::Value::as_f64) else {
            continue;
        };
        let clamped = prob.clamp(0.0, 1.0);
        sum += clamped;
        total += 1;
        if clamped < 0.45 {
            low += 1;
        }
    }
    if total == 0 {
        (0.0, 0.0)
    } else {
        (sum / total as f64, low as f64 / total as f64)
    }
}

fn segment_confidence_score(segment: &WhisperConfidenceSegment) -> f64 {
    let logprob_confidence = segment.avg_logprob.exp().clamp(0.0, 1.0) * 0.55;
    let word_confidence = segment.word_prob_mean.clamp(0.0, 1.0) * 0.45;
    let no_speech_penalty = segment.no_speech_prob * 0.45;
    let compression_penalty = ((segment.compression_ratio - 2.0).max(0.0) / 2.0).min(0.35);
    let low_word_penalty = (segment.low_word_prob_ratio * 0.35).min(0.25);
    (logprob_confidence + word_confidence
        - no_speech_penalty
        - compression_penalty
        - low_word_penalty)
        .clamp(0.0, 1.0)
}

fn segment_is_suspicious(segment: &WhisperConfidenceSegment) -> bool {
    let weak_logprob = segment.avg_logprob <= -1.25;
    let weak_words = segment.word_prob_mean > 0.0
        && (segment.word_prob_mean < 0.45 || segment.low_word_prob_ratio > 0.55);
    segment.compression_ratio > 2.4
        || (segment.no_speech_prob > 0.70 && weak_logprob)
        || segment.avg_logprob <= -1.60
        || weak_words
}

pub(super) fn collect_low_confidence_cue_spans(
    cues: &[SubtitleCue],
    confidence: &[Option<CueAsrConfidence>],
    profile: QualityProfile,
) -> Vec<LowConfidenceCueSpan> {
    if cues.is_empty() || cues.len() != confidence.len() {
        return Vec::new();
    }

    let floor = match profile {
        QualityProfile::Fast => 0.42,
        QualityProfile::Balanced => 0.50,
        QualityProfile::Strict => 0.56,
    };

    let mut spans = Vec::<LowConfidenceCueSpan>::new();
    let mut idx = 0usize;
    while idx < cues.len() {
        if !is_low_confidence_entry(confidence[idx], floor) {
            idx += 1;
            continue;
        }

        let start = idx;
        let mut end = idx + 1;
        let mut non_low_budget = 1usize;
        while end < cues.len() {
            if is_low_confidence_entry(confidence[end], floor) {
                non_low_budget = 1;
                end += 1;
                continue;
            }
            if non_low_budget == 0 {
                break;
            }
            non_low_budget -= 1;
            end += 1;
        }
        while end > start && !is_low_confidence_entry(confidence[end - 1], floor) {
            end -= 1;
        }
        if end <= start {
            idx += 1;
            continue;
        }

        let score = mean_confidence_score(&confidence[start..end]).unwrap_or(0.0);
        spans.push(LowConfidenceCueSpan {
            start,
            end,
            score,
            floor,
        });
        idx = end;
    }
    spans
}

fn is_low_confidence_entry(entry: Option<CueAsrConfidence>, floor: f64) -> bool {
    let Some(entry) = entry else {
        return false;
    };
    entry.score < floor
        || entry.suspicious
        || entry.avg_logprob <= -1.25
        || (entry.word_prob_mean > 0.0 && entry.word_prob_mean < 0.42)
        || entry.low_word_prob_ratio > 0.60
        || (entry.no_speech_prob > 0.65 && entry.score < floor + 0.08)
}

pub(super) fn mean_confidence_score(confidence: &[Option<CueAsrConfidence>]) -> Option<f64> {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for entry in confidence {
        let Some(entry) = entry else {
            continue;
        };
        let compression_penalty = ((entry.compression_ratio - 2.2).max(0.0) / 2.2).min(0.20);
        let word_penalty = (entry.low_word_prob_ratio * 0.20).min(0.15);
        sum += (entry.score - compression_penalty - word_penalty).clamp(0.0, 1.0);
        count += 1;
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}
