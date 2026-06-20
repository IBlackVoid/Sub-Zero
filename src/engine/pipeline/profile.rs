use crate::engine::transcribe::QualityProfile;

pub(super) fn default_mt_batch_for_profile(profile: QualityProfile) -> usize {
    match profile {
        QualityProfile::Fast => 32,
        QualityProfile::Balanced => 24,
        QualityProfile::Strict => 16,
    }
}

pub(super) fn default_mt_tokens_for_profile(profile: QualityProfile) -> usize {
    match profile {
        QualityProfile::Fast => 8_192,
        QualityProfile::Balanced => 6_144,
        QualityProfile::Strict => 4_096,
    }
}

pub(super) fn default_mt_oom_retries_for_profile(profile: QualityProfile) -> usize {
    match profile {
        QualityProfile::Fast => 1,
        QualityProfile::Balanced => 2,
        QualityProfile::Strict => 3,
    }
}

/// Base beam width per profile — the starting rung for the escalation ladder.
pub(super) fn default_mt_beam_for_profile(profile: QualityProfile) -> usize {
    match profile {
        QualityProfile::Fast => 2,
        QualityProfile::Balanced => 4,
        QualityProfile::Strict => 8,
    }
}
