use super::paths::checkpoint_dir_for;
use crate::engine::srt::SubtitleCue;
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Default, Clone)]
pub(super) struct RelationshipGraphStats {
    pub(super) node_count: usize,
    pub(super) edge_count: usize,
    pub(super) utterances_labeled: usize,
}

pub(super) fn write_relationship_graph_sidecar(
    input: &Path,
    source_cues: &[SubtitleCue],
    speakers: &[Option<String>],
) -> Result<(Option<PathBuf>, RelationshipGraphStats), String> {
    if source_cues.is_empty() || speakers.is_empty() || speakers.len() != source_cues.len() {
        return Ok((None, RelationshipGraphStats::default()));
    }

    let mut utterances_labeled = 0usize;
    let mut node_counts = HashMap::<String, usize>::new();
    let mut node_display = HashMap::<String, HashMap<String, usize>>::new();
    let mut node_features = HashMap::<String, NodeFeatures>::new();

    let mut edges = HashMap::<(String, String), usize>::new();
    let mut last = None::<String>;

    for (cue, speaker) in source_cues.iter().zip(speakers.iter()) {
        let Some(raw) = speaker.as_ref() else {
            continue;
        };
        let id = normalize_speaker_id(raw);
        if id.is_empty() {
            continue;
        }
        utterances_labeled += 1;

        *node_counts.entry(id.clone()).or_insert(0) += 1;
        node_display
            .entry(id.clone())
            .or_default()
            .entry(raw.clone())
            .and_modify(|v| *v += 1)
            .or_insert(1);

        node_features
            .entry(id.clone())
            .or_default()
            .observe_text(&cue.text);

        if let Some(prev) = last.as_ref() {
            if prev != &id {
                *edges.entry((prev.clone(), id.clone())).or_insert(0) += 1;
            }
        }
        last = Some(id);
    }

    if node_counts.is_empty() {
        return Ok((None, RelationshipGraphStats::default()));
    }

    let nodes = node_counts
        .iter()
        .map(|(id, count)| {
            let display = node_display
                .get(id)
                .and_then(most_common_key)
                .unwrap_or_else(|| id.clone());
            let features = node_features.get(id).cloned().unwrap_or_default();
            json!({
                "id": id,
                "display": display,
                "utterances": count,
                "features": features.as_json(),
            })
        })
        .collect::<Vec<_>>();

    let edges_json = edges
        .iter()
        .map(|((from, to), weight)| {
            json!({
                "from": from,
                "to": to,
                "weight": weight,
            })
        })
        .collect::<Vec<_>>();

    let payload = json!({
        "version": "1.0",
        "kind": "relationship-graph",
        "source_file": input.display().to_string(),
        "nodes": nodes,
        "edges": edges_json,
    });

    let dir = checkpoint_dir_for(input)?;
    let out_path = dir.join("relationship_graph.json");
    let serialized = serde_json::to_string_pretty(&payload)
        .map_err(|e| format!("{} serialize relationship graph: {e}", out_path.display()))?;
    std::fs::write(&out_path, serialized).map_err(|e| format!("{}: {e}", out_path.display()))?;

    let stats = RelationshipGraphStats {
        node_count: node_counts.len(),
        edge_count: edges.len(),
        utterances_labeled,
    };

    Ok((Some(out_path), stats))
}

#[derive(Debug, Default, Clone)]
struct NodeFeatures {
    cues: usize,
    exclaims: usize,
    questions: usize,
    polite_markers: usize,
}

impl NodeFeatures {
    fn observe_text(&mut self, text: &str) {
        self.cues += 1;
        if text.contains('!') || text.contains('！') {
            self.exclaims += 1;
        }
        if text.contains('?') || text.contains('？') {
            self.questions += 1;
        }
        // Lightweight register heuristic for Japanese.
        if text.contains("です") || text.contains("ます") {
            self.polite_markers += 1;
        }
    }

    fn as_json(&self) -> serde_json::Value {
        let denom = self.cues.max(1) as f64;
        json!({
            "cues": self.cues,
            "exclaim_ratio": self.exclaims as f64 / denom,
            "question_ratio": self.questions as f64 / denom,
            "polite_ratio": self.polite_markers as f64 / denom,
        })
    }
}

fn most_common_key(map: &HashMap<String, usize>) -> Option<String> {
    map.iter().max_by_key(|(_, v)| *v).map(|(k, _)| k.clone())
}

fn normalize_speaker_id(label: &str) -> String {
    let mut out = String::with_capacity(label.len());
    let mut last_was_sep = false;
    for ch in label.chars() {
        let lowered = ch.to_ascii_lowercase();
        if lowered.is_ascii_alphanumeric() {
            out.push(lowered);
            last_was_sep = false;
            continue;
        }
        if matches!(lowered, '_' | '-' | ' ') && !last_was_sep && !out.is_empty() {
            out.push('_');
            last_was_sep = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    out
}
