# Music generation

[Fixed vocal to accompaniment](vocal2accomp/README.md) and [lyrics to song](lyrics2song/README.md) have different input/output contracts and must not share an evaluation table without stating those differences. Both should report the trained reward, an independent metric and human listening separately. A combined reward is not evidence that each component works alone.
