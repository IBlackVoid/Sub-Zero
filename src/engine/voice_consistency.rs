use crate::engine::srt::SubtitleCue;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VoiceVector {
    pub contraction_ratio: f32,
    pub politeness_score: f32,
    pub first_person_ratio: f32,
    pub avg_sentence_len: f32,
    pub question_ratio: f32,
    pub interjection_ratio: f32,
}

impl VoiceVector {
    pub fn zero() -> Self {
        Self {
            contraction_ratio: 0.0,
            politeness_score: 0.0,
            first_person_ratio: 0.0,
            avg_sentence_len: 0.0,
            question_ratio: 0.0,
            interjection_ratio: 0.0,
        }
    }

    pub fn extract(text: &str) -> Self {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Self::zero();
        }
        let words: Vec<String> = trimmed
            .split_whitespace()
            .map(|w| {
                w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
                    .to_ascii_lowercase()
            })
            .filter(|w| !w.is_empty())
            .collect();
        let total = words.len().max(1) as f32;

        let contractions = words
            .iter()
            .filter(|w| w.contains('\'') && CONTRACTION_TAILS.iter().any(|t| w.ends_with(t)))
            .count() as f32;

        let mut politeness = 0.0_f32;
        for w in &words {
            if POLITE_TOKENS.iter().any(|t| t == w) {
                politeness += 1.0;
            } else if INFORMAL_TOKENS.iter().any(|t| t == w) {
                politeness -= 1.0;
            }
        }
        let politeness_score = (politeness / total).clamp(-1.0, 1.0);

        let pronouns = words
            .iter()
            .filter(|w| ALL_PRONOUNS.contains(&w.as_str()))
            .count() as f32;
        let first_person = words
            .iter()
            .filter(|w| FIRST_PERSON.contains(&w.as_str()))
            .count() as f32;
        let first_person_ratio = if pronouns > 0.0 {
            first_person / pronouns
        } else {
            0.0
        };

        let sentences = sentence_count(trimmed) as f32;
        let avg_sentence_len = total / sentences.max(1.0);

        let questions = trimmed
            .split_terminator(['.', '!', '?'])
            .filter(|s| !s.trim().is_empty())
            .filter(|s| s.contains('?'))
            .count() as f32;
        let question_ratio = if sentences > 0.0 {
            questions / sentences
        } else {
            0.0
        };

        let interjections = words
            .iter()
            .filter(|w| CASUAL_INTERJECTIONS.contains(&w.as_str()))
            .count() as f32;
        let interjection_ratio = interjections / total;

        Self {
            contraction_ratio: (contractions / total).clamp(0.0, 1.0),
            politeness_score,
            first_person_ratio,
            avg_sentence_len,
            question_ratio,
            interjection_ratio,
        }
    }

    pub fn squared_distance(&self, other: &Self) -> f32 {
        let d1 = self.contraction_ratio - other.contraction_ratio;
        let d2 = self.politeness_score - other.politeness_score;
        let d3 = self.first_person_ratio - other.first_person_ratio;
        let d4 = (self.avg_sentence_len - other.avg_sentence_len) / 8.0;
        let d5 = self.question_ratio - other.question_ratio;
        let d6 = self.interjection_ratio - other.interjection_ratio;
        d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 + d5 * d5 + d6 * d6
    }

    pub fn distance(&self, other: &Self) -> f32 {
        self.squared_distance(other).sqrt()
    }

    pub fn update_mean(&mut self, sample: &Self, n: u32) {
        let n_f = n as f32;
        let alpha = 1.0 / (n_f + 1.0);
        self.contraction_ratio += alpha * (sample.contraction_ratio - self.contraction_ratio);
        self.politeness_score += alpha * (sample.politeness_score - self.politeness_score);
        self.first_person_ratio += alpha * (sample.first_person_ratio - self.first_person_ratio);
        self.avg_sentence_len += alpha * (sample.avg_sentence_len - self.avg_sentence_len);
        self.question_ratio += alpha * (sample.question_ratio - self.question_ratio);
        self.interjection_ratio += alpha * (sample.interjection_ratio - self.interjection_ratio);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VoicePriors {
    #[serde(default)]
    pub priors: HashMap<String, SpeakerPrior>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerPrior {
    pub samples: u32,
    pub mean: VoiceVector,
}

#[derive(Debug, Clone, Serialize)]
pub struct VoiceDeviationReport {
    pub cue_index: usize,
    pub speaker: String,
    pub samples_in_prior: u32,
    pub deviation: f32,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct VoiceConsistencyStats {
    pub cues_scored: usize,
    pub speakers_observed: usize,
    pub mean_deviation: f32,
    pub p95_deviation: f32,
    pub max_deviation: f32,
}

impl VoicePriors {
    pub fn load_default() -> Self {
        match Self::path() {
            Some(p) if p.is_file() => std::fs::read_to_string(&p)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default(),
            _ => Self::default(),
        }
    }

    pub fn score_and_update(&mut self, speaker_key: &str, sample: &VoiceVector) -> Option<f32> {
        let entry = self
            .priors
            .entry(speaker_key.to_string())
            .or_insert_with(|| SpeakerPrior {
                samples: 0,
                mean: VoiceVector::zero(),
            });
        let dev = if entry.samples > 0 {
            Some(sample.distance(&entry.mean))
        } else {
            None
        };
        entry.mean.update_mean(sample, entry.samples);
        entry.samples += 1;
        dev
    }

    pub fn process_batch(
        &mut self,
        cues: &[SubtitleCue],
        speakers: &[Option<String>],
    ) -> (Vec<VoiceDeviationReport>, VoiceConsistencyStats) {
        let mut reports = Vec::new();
        let mut deviations = Vec::new();
        let n = cues.len().min(speakers.len());
        let mut observed = std::collections::HashSet::<String>::new();
        for i in 0..n {
            let Some(speaker) = speakers[i].as_ref() else {
                continue;
            };
            let key = normalize_speaker_id(speaker);
            if key.is_empty() {
                continue;
            }
            observed.insert(key.clone());
            let v = VoiceVector::extract(&cues[i].text);
            let prior_samples = self.priors.get(&key).map(|p| p.samples).unwrap_or(0);
            let dev = self.score_and_update(&key, &v);
            if let Some(d) = dev {
                deviations.push(d);
                reports.push(VoiceDeviationReport {
                    cue_index: cues[i].index,
                    speaker: key,
                    samples_in_prior: prior_samples,
                    deviation: d,
                });
            }
        }

        let stats = if deviations.is_empty() {
            VoiceConsistencyStats {
                cues_scored: 0,
                speakers_observed: observed.len(),
                ..Default::default()
            }
        } else {
            deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = deviations.len();
            let sum: f32 = deviations.iter().sum();
            let mean = sum / n as f32;
            let p95_idx = ((n as f32 * 0.95) as usize).min(n - 1);
            let p95 = deviations[p95_idx];
            let max = *deviations.last().unwrap_or(&0.0);
            VoiceConsistencyStats {
                cues_scored: n,
                speakers_observed: observed.len(),
                mean_deviation: mean,
                p95_deviation: p95,
                max_deviation: max,
            }
        };

        (reports, stats)
    }

    pub fn save(&self) -> std::io::Result<()> {
        let Some(p) = Self::path() else {
            return Ok(());
        };
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = p.with_extension("json.tmp");
        let s = serde_json::to_string_pretty(self).unwrap_or_default();
        std::fs::write(&tmp, s)?;
        std::fs::rename(&tmp, &p)?;
        Ok(())
    }

    pub fn path() -> Option<PathBuf> {
        if let Some(home) = std::env::var_os("VOIDEX_HOME") {
            return Some(PathBuf::from(home).join("voice_priors.json"));
        }
        let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
        Some(
            PathBuf::from(home)
                .join(".voidex")
                .join("voice_priors.json"),
        )
    }

    #[allow(dead_code)]
    pub fn load_from(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }
}

fn sentence_count(text: &str) -> usize {
    let raw = text
        .split_terminator(['.', '!', '?'])
        .filter(|s| !s.trim().is_empty())
        .count();
    raw.max(1)
}

fn normalize_speaker_id(s: &str) -> String {
    s.trim().to_ascii_lowercase()
}

const CONTRACTION_TAILS: &[&str] = &["'s", "'re", "'ve", "'ll", "'m", "'d", "n't", "'em"];

const POLITE_TOKENS: &[&str] = &[
    "please", "sir", "ma'am", "thank", "thanks", "kindly", "pardon", "excuse", "would", "could",
    "may",
];

const INFORMAL_TOKENS: &[&str] = &[
    "yeah", "yo", "dude", "bro", "ain't", "gonna", "wanna", "gotta", "lemme", "kinda", "sorta",
    "nah", "yep", "nope", "huh",
];

const FIRST_PERSON: &[&str] = &["i", "me", "my", "mine", "myself", "we", "us", "our", "ours"];

const ALL_PRONOUNS: &[&str] = &[
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "you", "your", "yours",
    "yourself", "he", "she", "it", "they", "him", "her", "them", "their", "theirs", "his", "hers",
    "its",
];

const CASUAL_INTERJECTIONS: &[&str] = &[
    "yeah", "oh", "huh", "well", "um", "uh", "wow", "hey", "ah", "ouch", "ugh", "hmm", "okay", "ok",
];

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(text: &str) -> SubtitleCue {
        SubtitleCue {
            index: 1,
            timing: "00:00:00,000 --> 00:00:01,000".to_string(),
            text: text.to_string(),
        }
    }

    #[test]
    fn extract_zero_for_empty_text() {
        let v = VoiceVector::extract("");
        assert_eq!(v.contraction_ratio, 0.0);
        assert_eq!(v.politeness_score, 0.0);
    }

    #[test]
    fn formal_speaker_has_positive_politeness() {
        let v = VoiceVector::extract("Please, sir, would you kindly excuse me.");
        assert!(v.politeness_score > 0.3);
    }

    #[test]
    fn informal_speaker_has_negative_politeness() {
        let v = VoiceVector::extract("Yeah, dude, ain't no way.");
        assert!(v.politeness_score < -0.3);
    }

    #[test]
    fn contraction_ratio_picks_up_apostrophes() {
        let v = VoiceVector::extract("I'm not sure I'd want to.");
        assert!(v.contraction_ratio > 0.2);
    }

    #[test]
    fn first_person_dominates_when_self_focused() {
        let v = VoiceVector::extract("I think I should go. I want to leave.");
        assert!(v.first_person_ratio > 0.9);
    }

    #[test]
    fn distance_increases_with_register_gap() {
        let formal = VoiceVector::extract("Pardon me, sir, but I would prefer to wait.");
        let casual = VoiceVector::extract("Yeah, nah, I'm gonna chill, dude.");
        let same = VoiceVector::extract("Pardon me, sir, but I would also like to wait.");
        assert!(formal.distance(&casual) > formal.distance(&same));
    }

    #[test]
    fn priors_track_running_mean() {
        let mut p = VoicePriors::default();
        let cues = vec![
            cue("I would prefer the formal option, thank you."),
            cue("Could you kindly explain the situation, sir?"),
            cue("Please excuse my delay; I would like to assist."),
        ];
        let speakers = vec![
            Some("alice".to_string()),
            Some("alice".to_string()),
            Some("alice".to_string()),
        ];
        let (reports, stats) = p.process_batch(&cues, &speakers);
        assert_eq!(reports.len(), 2);
        assert_eq!(stats.cues_scored, 2);
        assert_eq!(stats.speakers_observed, 1);
        assert_eq!(p.priors.get("alice").unwrap().samples, 3);
    }

    #[test]
    fn priors_distinguish_speakers() {
        let mut p = VoicePriors::default();
        let cues = vec![
            cue("Pardon me, would you kindly help?"),
            cue("Yeah dude, no way, ain't happening!"),
            cue("I would prefer the formal option."),
            cue("Yo, that's wack, lemme tell ya."),
        ];
        let speakers = vec![
            Some("alice".to_string()),
            Some("bob".to_string()),
            Some("alice".to_string()),
            Some("bob".to_string()),
        ];
        p.process_batch(&cues, &speakers);
        let alice = p.priors.get("alice").unwrap();
        let bob = p.priors.get("bob").unwrap();
        assert!(alice.mean.politeness_score > 0.0);
        assert!(bob.mean.politeness_score < 0.0);
    }

    #[test]
    fn priors_save_load_round_trip() {
        let tmp = std::env::temp_dir().join(format!(
            "voidex_voice_test_{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let mut p = VoicePriors::default();
        let v = VoiceVector::extract("Hello, friend, how are you?");
        p.score_and_update("alice", &v);
        let s = serde_json::to_string(&p).unwrap();
        std::fs::write(&tmp, s).unwrap();
        let p2 = VoicePriors::load_from(&tmp);
        assert_eq!(p2.priors.get("alice").unwrap().samples, 1);
        let _ = std::fs::remove_file(&tmp);
    }
}
