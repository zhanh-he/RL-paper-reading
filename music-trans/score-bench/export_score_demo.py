"""Export attributed ASAP reference and actual model score excerpts for inspection."""
import argparse
import json
import shutil
from pathlib import Path
import music21
import verovio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.data / "manifest.json").read_text())
    examples = []
    for identity in ["test-02", "test-07"]:
        item = next(s for s in manifest["samples"] if s["id"] == identity)
        target = args.output / identity
        target.mkdir()
        sources = {"reference": args.data / item["files"]["xml_score"]["path"],
            "baseline": args.data / "duet-baseline" / identity / "score.musicxml",
            "sft50": args.data / "duet-duration-sft50" / identity / "score.musicxml",
            "grpo50": args.data / "duet-duration-grpo50" / identity / "score.musicxml"}
        for name, source in sources.items():
            shutil.copy2(source, target / f"{name}-full.musicxml")
            score = music21.converter.parse(source)
            excerpt = score.measures(1, 4)
            excerpt.metadata = music21.metadata.Metadata()
            excerpt.metadata.title = f"{item['work']} · {name} · first 4 measures"
            xml = target / f"{name}-excerpt.musicxml"
            excerpt.write("musicxml", fp=str(xml))
            toolkit = verovio.toolkit()
            toolkit.setOptions({"pageWidth": 1600, "pageHeight": 900, "adjustPageHeight": True,
                "scale": 45, "header": "none", "footer": "none", "breaks": "auto"})
            if not toolkit.loadData(xml.read_text()): raise RuntimeError(f"Cannot render {xml}")
            (target / f"{name}.svg").write_text(toolkit.renderToSVG(1))
        examples.append({"id": identity, "work": item["work"], "note": "Post-hoc illustrative improved/worsened work, not representative sampling. First four notated measures shown; full-score metrics and complete MusicXML remain available."})
    (args.output / "examples.json").write_text(json.dumps(examples, indent=2) + "\n")


if __name__ == "__main__": main()
