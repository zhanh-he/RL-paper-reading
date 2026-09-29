# Presentation status: 30 September 2026

Use the [experiment replay](../platform/site/demos/index.html) to play the
public, original/synthetic samples. The labels below describe what was
actually run; a blank GRPO stage is not an implied positive result.

## Five-minute walk-through

1. **Choral is the central result.** On 30 held-out songs, ChoralStream's
   frame-head GRPO raised its *frame-head* F1 from 0.1378 to 0.3033 after
   1,152 optimizer updates. A supervised BCE comparison reached 0.4541.
   Critically, the full-song event decoder does not use this head: the
   independently decoded MIDI is identical before/after. Its pooled 50 ms
   pitch-only onset F1 is 0.1063 and onset+offset F1 is 0.0234 on both sides;
   SATB-track-aware values are 0.0661 and 0.0145. Do not present the frame-head
   result as a transcription improvement. See the [event receipt](../music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/note_metrics_50ms.json)
   and [public synthetic replay](../music-trans/choral-singing/rl/grpo/runs/2026-09-30-public-synthetic/receipt.json).
2. **MuScriptor is a baseline, not GRPO.** Its official online demo converted
   the same original 10.2-second SATB synthesis into 16 piano notes in one
   unlabeled track. Against the 32-note pitch reference, 50 ms onset F1 is
   0.6667 and onset+offset F1 is 0.5000; SATB track-aware F1 is undefined.
   This cannot be directly equated to ChoralStream's SATB score or a local
   post-training result. The public weights remain gated for our account.
   [Receipt](../music-trans/multi-inst/models/muscriptor/muscriptor_receipt.json).
3. **Lyrics-to-song has real but negative early GRPO comparisons.** YuE2-3B
   SongEval-GRPO was measured at 0/1/5/50/100 updates on three fixed held-out
   prompts; the corresponding SongEval means were
   3.8403/3.6030/3.6160/3.5704/3.6063. The 100-update first held-out clip
   has peak 0.851 and zero near-full-scale samples. Muse-0.6b SongEval one-step pilot moved one
   held-out SongEval score 3.5585 to 3.3330. A separate online MuseCritic
   one-step run on Gadi completed; its held-out MuseCritic mean fell 3.0781 to
   2.5313, while peak amplitude rose 0.197 to 0.422 without full-scale
   clipping. These are *not* quality gains or a reward-hacking finding.
   [MuseCritic receipt](../music-gen/lyrics2song/rl/grpo/runs/2026-09-30-musecritic-heldout/receipt.json).
   A separate 50-step MuseCritic run is active on Gadi with automatic 25/50
   held-out replay after training; no result is claimed for those steps yet.
4. **SongEval vulnerability is conditional.** In a controlled 47-clip
   perturbation audit, safe gain increased mean score by 0.0430; full-scale
   hard clipping by 0.0951 relative to normalized reference. After matching
   RMS, hard clipping had no clear mean advantage over clean gain (-0.0020,
   paired 95% interval [-0.0057, 0.0015]). A 6 kHz bandlimit/upsample
   control lost 0.1828. This supports loudness sensitivity, but does not
   establish the cause of any private model's clipping. No company samples
   or checkpoints are in this repository. [Audit receipt](../music-gen/lyrics2song/rewards/audits/2026-09-29/perturbation_47clips.json).
5. **Vocal-to-accompaniment is inference only.** ACE-Step 1.5 returned a
   full-mix completion and AnyAccomp returned an isolated accompaniment for
   the same original 16-second synthetic vocal guide. AnyAccomp coverage is
   75.9% and beat-v2 F1 is 0.368 on this single input; neither is a GRPO
   delta. A corresponding reference accompaniment and validated richness
   score are absent. [Receipt](../music-gen/vocal2accomp/models/anyaccomp/receipt.json).

## Next experiment, not an already observed result

For transcription, put the trainable policy on the **event decoder** and use
reference-backed 50 ms onset and onset+offset F1 as primary rewards; report
pitch-only and SATB-track-aware variants, and use frame F1, note continuity,
four-part occupancy, and voice-range violations as diagnostics or guardrails.
Four-part occupancy alone invites note flooding. For generation, use fixed
held-out prompts and blind A/B listening alongside each reward and clipping
guardrail before any claim that 100/300/1000 steps are underfit/goodfit/overfit.
The small-data hypothesis is a research question, not a result of these pilots.

## Demo checklist

- In each view, play input/reference where available, then baseline and GRPO.
- In Lyrics replay, switch experiments; SongEval and MuseCritic score scales
  are separate. Switch A/B at matched playback position, then inspect the
  waveform, spectrogram, score dimensions and local listening form.
- In Choral GRPO, switch piano rolls among reference, ChoralStream baseline,
  ChoralStream frame-head GRPO and MuScriptor baseline. The identical
  ChoralStream MIDI is the finding, not a demo failure.
- If embedded audio controls fail, use the adjacent direct audio link.
