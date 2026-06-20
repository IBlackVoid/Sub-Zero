use crate::engine::srt::SubtitleCue;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const MIN_CONSISTENT_OCCURRENCES: usize = 3;
const MIN_NAME_LENGTH: usize = 2;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CharacterGlossary {
    #[serde(default)]
    pub variants: HashMap<String, String>,
    #[serde(default)]
    pub seen_variants: HashMap<String, Vec<String>>,
}

impl CharacterGlossary {
    pub fn load_default() -> Self {
        match Self::path() {
            Some(p) if p.is_file() => std::fs::read_to_string(&p)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default(),
            _ => Self::default(),
        }
    }

    #[allow(dead_code)]
    pub fn load_from(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn apply(&self, cues: &mut [SubtitleCue]) {
        if self.variants.is_empty() {
            return;
        }
        for cue in cues.iter_mut() {
            cue.text = rewrite_words(&cue.text, &self.variants);
        }
    }

    pub fn learn(&mut self, cues: &[SubtitleCue]) {
        let mut freq: HashMap<String, usize> = HashMap::new();
        for cue in cues.iter() {
            for word in cue.text.split_whitespace() {
                let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
                if clean.len() < MIN_NAME_LENGTH {
                    continue;
                }
                if !clean
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
                {
                    continue;
                }
                *freq.entry(clean.to_string()).or_insert(0) += 1;
            }
        }

        // Process candidates in a deterministic, frequency-ranked order.
        //
        // `HashMap` iteration order is randomized per run, so iterating `freq`
        // directly made the cluster canonical non-deterministic AND let a
        // garbled spelling that happened to be visited first become the
        // canonical that every other variant — including the correct one — was
        // rewritten to. Ranking by descending frequency (then lexicographically
        // for stable tie-breaks) makes the most-attested spelling the canonical,
        // which is both reproducible and the better correctness prior.
        let mut ranked: Vec<(&String, usize)> = freq.iter().map(|(w, c)| (w, *c)).collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

        for (word, count) in ranked {
            if count < MIN_CONSISTENT_OCCURRENCES {
                continue;
            }
            if self.seen_variants.contains_key(word) {
                continue;
            }
            let mut absorbed = false;
            for canonical in self.seen_variants.keys().cloned().collect::<Vec<_>>() {
                if levenshtein(word, &canonical) <= 1 {
                    self.seen_variants
                        .entry(canonical.clone())
                        .or_default()
                        .push(word.clone());
                    self.variants.insert(word.clone(), canonical);
                    absorbed = true;
                    break;
                }
            }
            if !absorbed {
                self.seen_variants.insert(word.clone(), vec![word.clone()]);
                self.variants.insert(word.clone(), word.clone());
            }
        }
    }

    /// Canonical names learned so far (one per cluster), sorted for determinism.
    /// Fed to the LLM escalation rung so its rescued translations stay consistent
    /// with names the rest of the document already settled on — and, because the
    /// glossary persists across runs, with names learned in earlier episodes.
    pub fn canonical_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.seen_variants.keys().cloned().collect();
        names.sort();
        names
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

    fn path() -> Option<PathBuf> {
        if let Some(home) = std::env::var_os("VOIDEX_HOME") {
            return Some(PathBuf::from(home).join("character_glossary.json"));
        }
        let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
        Some(
            PathBuf::from(home)
                .join(".voidex")
                .join("character_glossary.json"),
        )
    }
}

fn rewrite_words(text: &str, variants: &HashMap<String, String>) -> String {
    let mut out = String::with_capacity(text.len());
    for word in text.split_inclusive(char::is_whitespace) {
        let core_start = word.find(|c: char| c.is_alphanumeric());
        let Some(start) = core_start else {
            out.push_str(word);
            continue;
        };
        let core_end = word
            .rfind(|c: char| c.is_alphanumeric())
            .map(|i| i + word[i..].chars().next().unwrap().len_utf8())
            .unwrap_or(word.len());
        let core = &word[start..core_end];
        if let Some(canonical) = variants.get(core) {
            out.push_str(&word[..start]);
            out.push_str(canonical);
            out.push_str(&word[core_end..]);
        } else {
            out.push_str(word);
        }
    }
    out
}

fn levenshtein(a: &str, b: &str) -> usize {
    let av: Vec<char> = a.chars().take(16).collect();
    let bv: Vec<char> = b.chars().take(16).collect();
    let m = av.len();
    let n = bv.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut cur: Vec<usize> = vec![0; n + 1];
    for i in 1..=m {
        cur[0] = i;
        for j in 1..=n {
            let cost = if av[i - 1] == bv[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[n]
}

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
    fn learn_promotes_repeated_capitalised_word() {
        let mut g = CharacterGlossary::default();
        g.learn(&[
            cue("Konatsu walked in."),
            cue("Yes, Konatsu said."),
            cue("Konatsu agreed."),
        ]);
        assert!(g.variants.contains_key("Konatsu"));
        assert_eq!(g.variants.get("Konatsu"), Some(&"Konatsu".to_string()));
    }

    #[test]
    fn learn_skips_below_threshold() {
        let mut g = CharacterGlossary::default();
        g.learn(&[cue("Konatsu walked in."), cue("Yes."), cue("OK.")]);
        assert!(!g.variants.contains_key("Konatsu"));
    }

    #[test]
    fn apply_canonicalises_known_variant() {
        let mut g = CharacterGlossary::default();
        g.variants
            .insert("Konozuka".to_string(), "Konatsu".to_string());
        let mut cues = vec![cue("Hello, Konozuka.")];
        g.apply(&mut cues);
        assert_eq!(cues[0].text, "Hello, Konatsu.");
    }

    #[test]
    fn apply_preserves_unknown_words() {
        let mut g = CharacterGlossary::default();
        g.variants
            .insert("Konozuka".to_string(), "Konatsu".to_string());
        let mut cues = vec![cue("The dog runs fast.")];
        g.apply(&mut cues);
        assert_eq!(cues[0].text, "The dog runs fast.");
    }

    #[test]
    fn learn_absorbs_levenshtein_one_variant() {
        let mut g = CharacterGlossary::default();
        g.learn(&[
            cue("Konatsu spoke."),
            cue("Konatsu smiled."),
            cue("Konatsu left."),
        ]);
        g.learn(&[
            cue("Konatsv replied."),
            cue("Then Konatsv whispered."),
            cue("Konatsv was right."),
        ]);
        assert_eq!(g.variants.get("Konatsv"), Some(&"Konatsu".to_string()));
    }

    #[test]
    fn learn_canonical_is_most_frequent_variant() {
        // The correct spelling "Konatsu" (4x) and a garble "Konatsv" (3x) are
        // within Levenshtein-1 and both clear the occurrence threshold. The
        // more frequent spelling must win as canonical, deterministically —
        // regardless of HashMap iteration order. Run several times to make a
        // non-deterministic regression statistically loud.
        for _ in 0..16 {
            let mut g = CharacterGlossary::default();
            g.learn(&[
                cue("Konatsu walked in."),
                cue("Konatsu smiled."),
                cue("Konatsu agreed."),
                cue("Konatsu left."),
                cue("Konatsv replied."),
                cue("Then Konatsv whispered."),
                cue("Konatsv was right."),
            ]);
            assert_eq!(g.variants.get("Konatsu"), Some(&"Konatsu".to_string()));
            assert_eq!(g.variants.get("Konatsv"), Some(&"Konatsu".to_string()));
        }
    }

    #[test]
    fn save_load_round_trip() {
        let tmp = std::env::temp_dir().join(format!(
            "voidex_glossary_test_{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let mut g = CharacterGlossary::default();
        g.variants.insert("Foo".to_string(), "Bar".to_string());
        let s = serde_json::to_string(&g).unwrap();
        std::fs::write(&tmp, s).unwrap();
        let g2 = CharacterGlossary::load_from(&tmp);
        assert_eq!(g2.variants.get("Foo"), Some(&"Bar".to_string()));
        let _ = std::fs::remove_file(&tmp);
    }
}
