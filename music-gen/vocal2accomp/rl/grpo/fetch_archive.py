"""Bounded parallel HTTP range download with resumable parts and MD5 verification."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import time
from urllib.request import Request, urlopen

URL = "https://zenodo.org/records/3338373/files/musdb18hq.zip"
SIZE = 22656664047
MD5 = "12d4f2ecd55245a4688754dd76363103"


def fetch(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "musdb18hq.zip"
    parts = root / "parts"
    parts.mkdir(exist_ok=True)
    chunk = (SIZE + 7) // 8
    if archive.exists() and 0 < archive.stat().st_size < chunk and not (parts / "000").exists():
        archive.replace(parts / "000")

    def part(index):
        start, end = index * chunk, min(SIZE, (index + 1) * chunk) - 1
        path = parts / f"{index:03d}"
        expected = end - start + 1
        for attempt in range(20):
            current = path.stat().st_size if path.exists() else 0
            if current == expected:
                return path
            if current > expected:
                raise ValueError("oversized download part")
            request = Request(URL, headers={"Range": f"bytes={start + current}-{end}",
                                           "User-Agent": "UWA-academic-audio-research/1.0"})
            try:
                with urlopen(request, timeout=90) as response:
                    wanted = f"bytes {start + current}-{end}/{SIZE}"
                    if response.status != 206 or response.headers.get("Content-Range") != wanted:
                        raise ValueError("Server did not honor exact range")
                    with path.open("ab") as handle:
                        while data := response.read(1024 * 1024):
                            handle.write(data)
                print(f"part={index} bytes={path.stat().st_size}", flush=True)
                if path.stat().st_size == expected:
                    return path
            except Exception as exc:
                print(f"part={index} retry={attempt} error={exc}", flush=True)
                time.sleep(min(60, 5 * (attempt + 1)))
        raise RuntimeError(f"Part {index} did not finish")

    if not archive.exists() or archive.stat().st_size != SIZE:
        with ThreadPoolExecutor(max_workers=8) as pool:
            files = list(pool.map(part, range(8)))
        temporary = root / "assembled.tmp"
        with temporary.open("wb") as dest:
            for path in files:
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        dest.write(block)
        temporary.replace(archive)
    digest = hashlib.md5()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != MD5:
        raise ValueError("Official archive checksum mismatch")
    print("OFFICIAL_MD5_OK", digest.hexdigest(), flush=True)


if __name__ == "__main__":
    import sys
    fetch(Path(sys.argv[1]))
