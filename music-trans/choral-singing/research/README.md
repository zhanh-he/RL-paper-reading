# Choral transcription: data, models, rewards

Source check: 13 Oct 2026. This is a public research map, not a new experiment or a claim that every linked dataset has been downloaded. The target is a mixture waveform to four named S/A/T/B MIDI note tracks. A pitch salience map, four F0 curves, a merged note track, and instrument-labelled MIDI are useful intermediate outputs, but none is interchangeable with part-aware note events.

## Dataset Protocol

| Source | What is actually supplied | Use / caveat |
| --- | --- | --- |
| [PawChorale v1.0.0](https://zenodo.org/records/21777039) / [browser](https://pawchorale.github.io/) | 200 works, 6.80 h master audio, 27.22 h isolated parts, 110,702 aligned vocal MIDI notes, MusicXML and rights manifests | Best new public candidate for part-aware notes. Parts vary; do not silently require four active voices. Rights are edition-specific and song/score overlap needs auditing before cross-dataset testing. |
| YouChorale / YouChorale-Pro as described in [PawCT](https://arxiv.org/abs/2610.09861) and our [event-head receipt](../rl/grpo/runs/2026-09-30-event-head/README.md) | Aligned choral note supervision used in the local experiment | Existing 16-song train / 8-song test, one 5.12 s clip per song, is a local protocol. Raw recordings, song IDs and weights are not republished here. Do not equate this split with the paper's table. |
| [Choral Singing Dataset](https://zenodo.org/records/2649950) | Three works, 16 close-mic singers, F0/note annotations and semi-synchronized per-section MIDI | Real voice adaptation and small alignment audit. The section MIDI needs timing checks against mixtures. |
| [ESMUC Choir Dataset](https://zenodo.org/records/5848990) | Three works, 12 singers, room mics plus close mics, manually corrected F0 and note annotations | Real-acoustic holdout; includes double-soprano/double-tenor scores and uneven voice counts. Do not collapse singers to four tracks without an explicit score-part map. |
| [Cantoría](https://zenodo.org/records/5851070) | Eleven SATB quartet songs, individual stems and mixtures, automatically estimated pYIN/CREPE F0 | Domain-shift/F0 test; automatic F0 is not a manually verified four-track MIDI note ground truth. |
| [Dagstuhl ChoirSet](https://www.audiolabs-erlangen.de/resources/MIR/2020-DagstuhlChoirSet) | Multitrack takes, aligned score, beat positions and extracted F0 | Score alignment and F0 evaluation; confirm performed note boundaries before treating the score as event-level labels. |
| [ChoralSynth](https://github.com/MTG/ChoralSynth) | Twenty curated synthetic choral songs created from score/MIDI | Controlled pretraining and error injection; verify particular audio/MIDI alignment and license per release. |
| [CocoChorales](https://magenta.tensorflow.org/datasets/cocochorales) | 1,400+ h of synthetic four-part *instrument* ensembles with source MIDI and F0 | Useful for symbolic four-part structure, not a real-singing benchmark. |

**Proposed next protocol, not yet run:** inspect rights; group by work, score edition, singer and recording session; hold out works before cropping or transposition; keep an active-part mask per window; validate MIDI time origin, tempo map, onsets/offsets and pitch against stems; train on strongly aligned SATB notes, optionally pretrain on synthetic structure, and keep weak F0/score alignment labels separate. Pick hyperparameters on a distinct validation set. Seal a same-domain work-level test and report CSD/ESMUC/Cantoría separately as transfer tests, with annotation strength stated. Check composer/work identity across corpora before calling a test unseen. A 5.12 s clip result is not a full-song result.

## Models

| Family | Primary work and release | Correct role |
| --- | --- | --- |
| Direct part-aware | [PawCT preprint](https://arxiv.org/abs/2610.09861), [demo/code entry](https://hanyu-meng.github.io/Paw_Choral_AMT_Demo/) | Part-specific onset, offset, frame and presence with range/ordered-continuity priors. Submitted to ICASSP 2027 as of this check; not described as accepted. It reports 0.225 macro part-aware note F1 at 50 ms on its YouChorale protocol versus adapted baseline 0.165. These are paper values, not our GRPO values. |
| Choral note, no named parts | [Yu and Duan, ISMIR 2024](https://ismir.net/conferences/ismir-2024/) | Choral note detection but merged note output. Add a voice assignment stage before comparing SATB track-aware scores. |
| General multitrack | [MT3, ICLR 2022](https://openreview.net/forum?id=iMSjopcOn0p) / [code](https://github.com/magenta/mt3); [YourMT3+](https://arxiv.org/abs/2407.04822) / [code](https://github.com/mimbres/YourMT3) | Transfer/inference baselines. Their instrument tracks are not named choir parts; document any adapter. MT3's official repository says training is not easily supported, so do not present it as a turnkey GRPO policy. |
| General multitrack, trainable candidate | [MuScriptor, ISMIR 2026](https://arxiv.org/abs/2607.08168) / [official code and gated weights](https://github.com/muscriptor/muscriptor) | Open code, CC BY-NC weights, decoder-only small/medium/large. The lab has a [medium token-head GRPO smoke](../../multi-inst/models/muscriptor/README.md), but its pitch-only short-excerpt metric is *not* SATB part-aware evidence. A voice-aware adapter and multi-song split are still required. |
| F0 / two-stage | [Cuesta et al., multi-F0, ISMIR 2020](https://arxiv.org/abs/2009.04172) / [code](https://github.com/helenacuesta/multif0-estimation-polyvocals); [Cuesta and Gomez, Voice Assignment, TISMIR 2022](https://doi.org/10.5334/tismir.121) / [code](https://github.com/helenacuesta/voas-vocal-quartets) | Late/Deep multi-F0 provides pitch salience; VoAS splits salience into four sources. Thresholding, note segmentation, timing alignment and SATB assignment checks are still needed before note-level scoring. The VoAS repository calls itself work in progress. |

The useful bridge to multi-instrument AMT is event modeling and track assignment, not direct score transfer. A model may correctly identify pitches yet put them on a piano track, swap S/A, or miss quiet Bass. Evaluate those errors separately.

## Rewards and evaluation

The [implemented reward](../rewards/README.md) uses reference MIDI, one-to-one matching, 50 ms onset tolerance, offset tolerance `max(50 ms, 20% of reference duration)`, 10 ms frame grid and 0.5 s active-voice windows. Combined weights: SATB onset F1 .25, complete-note F1 .25, frame F1 .15, active-voice F1 .10, minimum per-voice complete-note F1 .10, complete-note precision .10, and `1 - fragmentation` .05. These are *our* GRPO components, not PawCT's training loss. See [event-head run](../rl/grpo/runs/2026-09-30-event-head/README.md) and [counterexample audit](../rewards/audits/2026-09-30-satb-reward-counterexamples.json) for evidence.

| Single signal | What it catches | How to attack/test it |
| --- | --- | --- |
| Part-aware onset F1 | Note starts in the correct voice | Shorten every note without moving starts. |
| Part-aware onset+offset F1 | Complete note event and duration | Delete difficult quiet-part notes; inspect macro *and* per-part recall. |
| Frame F1 | Occupied pitch/time cells | Fragment sustained notes; occupancy can stay high. |
| Active-part coverage | Missing S/A/T/B when reference is active | Add wrong octave notes inside active parts. Never demand four voices during rests. |
| Precision / weak-part floor / continuity | Overfill, vanished Bass and choppy sustains | Precision-only can prefer deletion; continuity-only can prefer silence; minimum-voice term is noisy with few notes. |

For publication, report per-part and macro frame F1, part-aware onset F1 (50 ms), and part-aware complete-note F1 (50 ms onset, stated offset tolerance), with note count, precision/recall, active-part mask and swapped-part confusion. F0 accuracy or octave error from the two-stage pipeline belongs in a separate table, not in the note-F1 column. Hold reward curves, fixed-checkpoint test F1 and ACE-rendered subjective replay apart. The combined objective has demonstrated blind-spot checks, not a guarantee against reward hacking; keep single-component attacks and held-out comparisons visible.
