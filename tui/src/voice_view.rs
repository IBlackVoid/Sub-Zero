use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
struct VoiceVectorRaw {
    contraction_ratio: f32,
    politeness_score: f32,
    first_person_ratio: f32,
    avg_sentence_len: f32,
    question_ratio: f32,
    interjection_ratio: f32,
}

#[derive(Debug, Clone, Deserialize)]
struct SpeakerPriorRaw {
    samples: u32,
    mean: VoiceVectorRaw,
}

#[derive(Debug, Clone, Deserialize)]
struct VoicePriorsFile {
    #[serde(default)]
    priors: std::collections::HashMap<String, SpeakerPriorRaw>,
}

#[derive(Debug, Clone)]
pub struct SpeakerSignature {
    pub name: String,
    pub samples: u32,
    pub bars: [f32; 6],
}

pub fn priors_path() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("SUB_ZERO_HOME") {
        return Some(PathBuf::from(home).join("voice_priors.json"));
    }
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
    Some(
        PathBuf::from(home)
            .join(".sub-zero")
            .join("voice_priors.json"),
    )
}

pub fn load() -> Vec<SpeakerSignature> {
    let Some(path) = priors_path() else {
        return Vec::new();
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let Ok(file): Result<VoicePriorsFile, _> = serde_json::from_str(&text) else {
        return Vec::new();
    };
    let mut out: Vec<SpeakerSignature> = file
        .priors
        .into_iter()
        .map(|(name, prior)| {
            let politeness_norm = ((prior.mean.politeness_score + 1.0) / 2.0).clamp(0.0, 1.0);
            let sent_len_norm = (prior.mean.avg_sentence_len / 20.0).clamp(0.0, 1.0);
            SpeakerSignature {
                name,
                samples: prior.samples,
                bars: [
                    prior.mean.contraction_ratio.clamp(0.0, 1.0),
                    politeness_norm,
                    prior.mean.first_person_ratio.clamp(0.0, 1.0),
                    sent_len_norm,
                    prior.mean.question_ratio.clamp(0.0, 1.0),
                    prior.mean.interjection_ratio.clamp(0.0, 1.0),
                ],
            }
        })
        .collect();
    out.sort_by(|a, b| b.samples.cmp(&a.samples));
    out
}

pub fn bar_glyph(v: f32) -> char {
    crate::waveform::block_for(v)
}

pub const FEATURE_HEADER: &str = "ctr pol 1st len qst itj";
