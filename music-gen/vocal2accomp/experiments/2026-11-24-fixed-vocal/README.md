# Fixed-vocal accompaniment pilot

[Accompaniment Lab](https://zhanh-he.github.io/RL-paper-reading/demos/vocal-accompaniment/?session=pilot) · [Public run receipt](https://zhanh-he.github.io/RL-paper-reading/demos/vocal-accompaniment/audio/pilot/receipt.json)

StepAudio 3 Music §5.5 describes limitations from sampling one output per condition, excluding invalid generations from common subsets, and limited evidence on vocal-to-mix comparisons and fine-grained rhythmic adherence. This pilot retains failures, repeats style-conditioned generation and makes timing evidence inspectable at specific song positions. [Technical report §5.5](https://arxiv.org/html/2609.16034v1#S5.SS5)

The three Vocadito excerpts each have two LaDA-Band candidates in acoustic folk, sparse synth-pop and half-time R&B, and one dry-only AnyAccomp baseline. A separate authored-source check uses an excerpt of Kara Square’s “Hey, Are You Here?” and tests two acoustic folk renders, one sparse synth-pop render and one AnyAccomp baseline. These are accompaniment generation runs. Each listening mix combines the hash-verified dry input and the isolated model backing, unshifted at fixed gain. The downloadable receipt records source and output bindings, GPU timings, and the resolved environment failure. The StepAudio route has a validated request preview and zero live submissions because the authorized runtime had no API credential.

Kara Square published the dry stem under CC BY 3.0. The release includes the sample-accurate dry excerpt and generated derivatives. The artist’s companion mix corroborates the 123 BPM reference, with 14 of 14 downbeat pairs and +8 to +35 ms raw lag. The Vocadito dataset is CC BY 4.0, and its authors describe participant-selected original or public-domain compositions. Individual composition status is unavailable. The Vocadito bar positions are analyst-proposed navigation anchors. Independent aural review of both sources' reference bars is pending. [Kara Square stem](https://ccmixter.org/files/mindmapthat/60165) · [Vocadito](https://doi.org/10.5281/zenodo.5578807)

The clearest measured failure is Vocadito 33 acoustic folk seed 73. Its file is 30.04 s, with backing below the detector's −55 dBFS gate for 29.42 s. The matched style and input at seed 29 have backing activity at the start. Vocadito 6 half-time R&B seed 73 has a 2.26 s initial gap below the gate. This may be a musically acceptable choice and awaits review.

The blind interface randomizes order within the same dry input and style. It records rhythmic fit and personal preference separately, retains ties and skips, and requires a listener’s agreement for reuse of a directional judgment. There are zero collected human judgments and no evidence of a listener preferring a rhythmic deviation. Hold out entire compositions for any independent follow-up evaluation.

[Inference and API replay](INFERENCE.md) · [Timing, pairing and controls](TIMING.md)
