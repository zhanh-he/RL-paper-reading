"""A bounded composition interface for an explicitly calibrated naturalness head.

Does not change existing trainers or their KL coefficient. This example explicitly
rescales the existing reward to [-1, 1]; adapting an existing trainer needs review.
Missing calibration/head output raises instead of silently inventing a reward.
"""
from dataclasses import dataclass
import math


def scale(value, p05, p95):
    if not all(math.isfinite(v) for v in (value,p05,p95)) or p95<=p05:
        raise ValueError('Finite values and nondegenerate frozen calibration required')
    return max(-1.,min(1.,2*(value-p05)/(p95-p05)-1))


def aggregate_windows(values):
    """10s windows, 5s hop; retain poor regions without optimizing a single minimum."""
    import numpy as np
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Need finite per-window values; do not drop failed windows')
    return float(.7*x.mean()+.3*np.quantile(x,.2))


@dataclass(frozen=True)
class CalibratedReward:
    existing_p05: float
    existing_p95: float
    natural_p05: float
    natural_p95: float
    pq_p05: float
    pq_p95: float
    weight: float = .05
    quality_floor: float = -.5
    quality_penalty: float = .1

    def compose(self, *, existing, naturalness_logit, pq, valid, activity_ok=True,
                duration_ok=True, clipping_fraction=0., shared_guard_already_applied=True):
        if not 0<=self.weight<=.1: raise ValueError('Exploratory naturalness weight must be 0..0.1')
        if not math.isfinite(clipping_fraction) or not 0<=clipping_fraction<=1:
            raise ValueError('Invalid clipping fraction')
        if not valid or not duration_ok or not activity_ok:
            return {'reward':-1.,'status':'rejected_invalid_or_task_coverage','naturalness_used':False}
        b=scale(existing,self.existing_p05,self.existing_p95)
        n=scale(naturalness_logit,self.natural_p05,self.natural_p95)
        q=scale(pq,self.pq_p05,self.pq_p95)
        # Existing guarded arms must not be penalized a second time. Enabling this
        # optional heuristic needs its own gain/intentional-distortion validation.
        clip_penalty=0. if shared_guard_already_applied else .1*min(1.,clipping_fraction/.01)
        reward=b+self.weight*min(n,q)-self.quality_penalty*max(0.,self.quality_floor-q)-clip_penalty
        return dict(reward=max(-1.,min(1.,reward)),unclipped_reward=reward,saturated=abs(reward)>1,
                    existing=b,naturalness=n,quality=q,
                    clip_penalty=clip_penalty,status='experimental_calibrated_composition',naturalness_used=True)


# Integration after scoring the exact decoded rollout (hash-bound):
# row = scorer.score(audio, sample_rate)
# reward = calibrated.compose(existing=old_reward, naturalness_logit=row['head_logit'],
#     pq=row['audiobox']['PQ'], valid=row['signal']['valid'],
#     activity_ok=task_specific_activity_check, duration_ok=expected_duration_check,
#     clipping_fraction=row['signal']['near_full_scale_fraction'],
#     shared_guard_already_applied=True)['reward']
# Inspect the caller: existing richness/instrument/arrangement already subtract a
# shared guard, whereas the repository Mean7 path currently does not. Do not copy
# this True flag without checking the chosen arm; choose one guard owner explicitly.
# Preserve old_reward, all raw scores, audio hash and calibration version in rollout receipt.
