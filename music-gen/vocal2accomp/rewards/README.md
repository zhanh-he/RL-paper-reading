# Vocal-to-accompaniment rewards

Keep simple implementations directly in this folder, not in three one-file subfolders:

| Intended file | What it should measure | Independent failure check |
| --- | --- | --- |
| `beat.py` | Timing fit to the **fixed original vocal**; Beat v2 is the requested first arm, v5 a version control | Repetition, drift, human rhythmic fit |
| `coverage.py` | Accompaniment activity in appropriate vocal sections | Noise beds, uninterrupted droning, missing sections |
| `richness.py` | Useful, coordinated arrangement layers | Unrelated layers, noise and harmonic clashes |
| `combine.py` | Calibrated composition of the three after single-arm tests | Floor on quality/coverage; no one score compensates for a hard failure |

Do not create these `.py` files as empty stubs. Migrate the actual, tested vocal2accomp definitions with their versioned dependencies and unit tests. Each reward should accept an explicit fixed-vocal reference, generated accompaniment and metadata, and return a scalar plus diagnostic components. `combine.py` should call the three modules and record their raw and normalized values; it must not hide them behind only one aggregate number.
