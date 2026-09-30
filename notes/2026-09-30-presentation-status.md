# Presentation status: 30 September 2026

[Open the experiment replay](../platform/site/demos/index.html). Playable
choral examples use an original synthetic SATB song and are separate from the
8-song held-out measurement. No company audio, weights or training receipts
are published.

## Five-minute walkthrough

1. **Choral transcription is the main result.** Event-level ChoralStream GRPO
   updates the MIDI pitch, voice, onset and duration heads. From 16 original
   training songs, one 5.12 s segment each, the combined 1000-step run changed
   8 disjoint test songs' SATB macro frame/onset/complete-note F1 from
   `0.2917/0.1105/0.0269` to `0.3857/0.1434/0.0419`. The strict note score
   remains low. At the **same 1000 requested steps**, none of seven
   single-reward arms beats the combination on any of these three metrics.
   [Protocol and complete ablation table](../music-trans/choral-singing/rl/grpo/runs/2026-09-30-event-head/README.md).
2. **Show why reward design matters, with qualified language.** Frame-only
   reaches frame F1 `0.3425` but complete-note F1 falls to `0.0144`; the
   combined run reaches `0.0419`. Coverage-only worsens onset F1 to `0.0899`.
   Continuity-only makes 192 effective updates; weak-voice only makes 1
   because most groups have no reward contrast. A single public song supplies
   matched ACE Studio baseline and 300-step reward-arm audio/MIDI; frame-only
   scores `0.749` frame F1 there but `0.000` complete-note F1. The synthetic
   short-note, octave-fill, silence, fragmentation and voice-swap audits are
   deliberately constructed counterexamples, **not** optimizer-discovered
   exploits. The data support complementary rewards in this pilot, not a
   universal anti-hacking guarantee.
3. **Use the paper-like SATB table correctly.** The demo presents S/A/T/B
   frame, 50 ms onset and onset+offset F1, macro averages, and VA rates next
   to the PawCT ICASSP 2027 draft Table 2. That paper's `Note` is onset-only;
   our extra complete-note column adds an offset tolerance of
   `max(50 ms, 20% reference duration)`. The paper evaluates original
   YouChorale and our pilot uses YouChorale-Pro short segments, so **do not
   compare their numerical scores directly**. Our onset/complete-note code
   agrees with `mir_eval` per SATB part on all public checkpoint MIDIs.
4. **Keep the earlier negative control visible.** A 1152-update frame-head
   GRPO raised frame-head F1 from `0.1378` to `0.3033` on 30 held-out songs,
   but the event decoder bypasses that head: full-song MIDI was unchanged.
   This is why the new event-head experiment is the transcription result.
5. **Other cases are narrower.** YuE2-3B SongEval-GRPO measured 0/1/5/50/100
   updates on three fixed prompts, with reward means
   `3.8403/3.6030/3.6160/3.5704/3.6063`, not an improvement. Muse/SongEval
   one-step went `3.5585 -> 3.3330` on one sample. MuScriptor-medium's local
   50-step token-head GRPO smoke moved one 5 s training song's pitch-only
   onset/complete-note F1 `0.179/0.123 -> 0.292/0.213`; a later anonymous
   8-song disjoint short-excerpt audit moved macro `0.043/0.037 ->
   0.109/0.103`. The audit was added after the smoke and lacks SATB labels;
   it is not a full-song benchmark. Muse/MuseCritic 25/50-step
   Gadi runs have paired replays, but stochastic MuCodec rendering prevents
   attributing a single audio-pair difference solely to training. A 47-clip
   SongEval perturbation audit found gain sensitivity, not a causal diagnosis
   of the private Huawei clipping observation. AnyAccomp and ACE-Step 1.5 are
   vocal-to-accompaniment inference baselines; no GRPO delta is claimed.

## Model gates and honest boundaries

- **MuScriptor:** the author's research permission and the current Hugging
  Face account's gated-weight access are different. Official small still
  returns HTTP 403, but an older authorized **medium** checkpoint was found
  and loaded from the team's offline benchmark cache. A 50-step token-head
  GRPO smoke with sampled MIDI, clipped ratios and frozen-reference KL
  completed on lab5090. Its same-song training gain and exploratory 8-song
  short-excerpt gain cannot establish robust full-song generalization. The
  hosted-demo output is one unlabeled piano track, not SATB transcription.
- **LaDA-Band:** the official Tencent code is cloned on lab5090. Its weights
  are also gated (HTTP 403 to the current account). The released diffusion
  inference/training code is not an on-policy GRPO adapter; implementing a
  trajectory likelihood/ratio path would be required after weight access.
  No LaDA baseline or GRPO result is in the demo.
- **No preset quality labels:** on the one public choral song, combined onset
  F1 is `0.133/0.195/0.306/0.123` at 0/100/300/1000 steps, whereas the
  held-out 8-song macro onset F1 peaks at 1000. The one-song 1000-step
  regression is a listening case, not proof of overfit or reward hacking.

## Demo order

Open Choral: play input and reference, then use the four-color SATB MIDI panels.
A is the frozen baseline, B switches among combined 0/100/300/1000, and C
switches among seven single-reward 300-step audio/MIDI replays. All use the
same ACE singers. The single held-out table contains combined 0/100/300/1000
and single-reward 300/1000 scores, with each F1 immediately followed by its
delta from frozen. The older frame-head control, per-part and PawCT tables,
and MuScriptor pilot are in the collapsed detail section. Open Rewards for
constructed counterexamples, then finish with YuE2/Muse and vocal baselines.
Do not label 100/300/1000 as underfit/goodfit/overfit without independent
listening and further splits. If an embedded player fails, use its direct link.
