"""Prepare or run one StepAudio 3 Music dry-vocal smoke test."""
import argparse
import base64
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo


BASE_URL = "https://api.stepfun.com/v1/audio/music"
MODEL_GUIDE = "https://platform.stepfun.com/docs/zh/guides/models/stepaudio-3-music"
API_REFERENCE = "https://platform.stepfun.com/docs/zh/api-reference/audio/music.md"
POLL_SECONDS = 5
SAMPLING = {
    "model_id": "stepaudio-3-music-preview",
    "task": "vocal_to_music",
    "response_format": "wav",
    "sample_rate": 48000,
    "lyrics_rewrite": False,
    "disable_caption_rewrite": True,
    "max_tokens": 16000,
    "temperature": 0.85,
    "top_k": 80,
    "top_p": 0.92,
    "repetition_penalty": 1.08,
}
TRANSIENT_QUERY_ERRORS = {"rate_limited", "service_unavailable", "internal"}


class NoRedirect(HTTPRedirectHandler):
    """Keep the authorization header on the documented API host."""

    def redirect_request(self, request, fp, code, msg, headers, url):
        return None


class StepHTTPError(Exception):
    def __init__(self, status_code, error_type, should_retry):
        super().__init__(f"HTTP {status_code} ({error_type})")
        self.status_code = status_code
        self.error_type = error_type
        self.should_retry = should_retry


class StepTaskFailed(Exception):
    """Terminal service-side failure, including HTTP-200 FAILED tasks."""


def awst_now():
    return datetime.now(ZoneInfo("Australia/Perth")).isoformat()


def save_json(path, value):
    scratch = path.with_suffix(".tmp")
    scratch.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    scratch.replace(path)


def post(action, payload, api_key, timeout):
    request = Request(
        f"{BASE_URL}/{action}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with build_opener(NoRedirect).open(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            body = json.load(exc)
            error_type = body.get("error", {}).get("type", "unknown")
        except (ValueError, AttributeError):
            error_type = "unknown"
        raise StepHTTPError(
            exc.code, error_type, exc.headers.get("x-should-retry") == "true"
        ) from None


def audio_metadata(path):
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels:format=duration",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(probe.stdout)
    audio, = result["streams"]
    return {
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "seconds": float(result["format"]["duration"]),
        "sample_rate": int(audio["sample_rate"]),
        "channels": audio["channels"],
    }


def run_remote(request, key, timeout, output, receipt):
    started = time.monotonic()
    deadline = started + timeout
    before_request = time.monotonic()
    submit = post("submit", request, key, min(45, timeout))
    receipt["submit_seconds"] = time.monotonic() - before_request
    receipt["task_id"] = submit["task_id"]
    save_json(output / "submit-response.json", submit)
    save_json(output / "receipt.json", receipt)

    while (remaining := deadline - time.monotonic()) > POLL_SECONDS:
        time.sleep(POLL_SECONDS)
        before_request = time.monotonic()
        try:
            reply = post("query", {"task_id": receipt["task_id"]}, key, min(45, remaining - POLL_SECONDS))
        except StepHTTPError as exc:
            receipt["polls"].append({
                "elapsed_seconds": time.monotonic() - started,
                "http_error": exc.status_code,
                "error_type": exc.error_type,
            })
            if exc.should_retry and exc.error_type in TRANSIENT_QUERY_ERRORS:
                continue
            raise
        except (URLError, TimeoutError):
            receipt["polls"].append({
                "elapsed_seconds": time.monotonic() - started,
                "error_type": "query_transport_failure",
            })
            continue

        status = reply.get("status")
        receipt["polls"].append({
            "elapsed_seconds": time.monotonic() - started,
            "request_seconds": time.monotonic() - before_request,
            "status": status,
        })
        response_metadata = {field: value for field, value in reply.items() if field != "audio"}
        save_json(output / "last-query-response.json", response_metadata)
        save_json(output / "receipt.json", receipt)
        if status == "FAILED":
            stage = reply.get("error", {}).get("stage", "unknown")
            raise StepTaskFailed(f"Task failed during {stage}")
        if status == "SUCCESS":
            audio = base64.b64decode(reply["audio"], validate=True)
            path = output / "generated-mix.wav"
            path.write_bytes(audio)
            receipt["output_audio"] = audio_metadata(path)
            receipt["output_audio"]["kind"] = "generated_mix"
            receipt["duration_difference_seconds"] = (
                receipt["output_audio"]["seconds"] - receipt["input_audio"]["seconds"]
            )
            receipt["sung_lyrics_review"] = "pending_manual_audio_review"
            receipt["status"] = "success"
            return
        if status not in {"PENDING", "RUNNING"}:
            raise StepTaskFailed(f"Unknown task status {status!r}")

    raise TimeoutError("No terminal task state before the polling deadline")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate locally without reading a key or making requests")
    mode.add_argument("--submit", action="store_true", help="Submit once using STEP_API_KEY from the environment")
    parser.add_argument("--vocal", type=Path, required=True)
    parser.add_argument("--lyrics-file", type=Path, required=True)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--private-output", type=Path, required=True, help="New directory outside the public repository")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= POLL_SECONDS or not args.caption.strip():
        parser.error("Provide a nonempty caption and a timeout longer than one poll interval.")
    lyrics = args.lyrics_file.read_text(encoding="utf-8").strip()
    if not lyrics:
        parser.error("Provide the sung lyrics in a nonempty UTF-8 file.")

    output = args.private_output.resolve()
    repo_root = Path(__file__).resolve().parents[4]
    if output.is_relative_to(repo_root):
        parser.error("Keep the private output directory outside the public repository.")
    os.umask(0o077)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    input_info = audio_metadata(args.vocal)
    input_bytes = args.vocal.read_bytes()
    if sha256(input_bytes).hexdigest() != input_info["sha256"]:
        raise RuntimeError("The vocal file changed while preparing the request.")
    request = dict(SAMPLING, caption=args.caption, lyrics=lyrics)
    receipt = {
        "schema_version": "stepfun-vocal-to-music-smoke-v1",
        "started_at_awst": awst_now(),
        "model_guide": MODEL_GUIDE,
        "api_reference": API_REFERENCE,
        "request_parameters": SAMPLING,
        "input_audio": input_info,
        "lyrics_sha256": sha256(lyrics.encode("utf-8")).hexdigest(),
        "caption_sha256": sha256(args.caption.encode("utf-8")).hexdigest(),
        "seed_controllable": False,
        "timeout_seconds": args.timeout_seconds,
        "poll_interval_seconds": POLL_SECONDS,
        "polls": [],
    }

    started = time.monotonic()
    try:
        if args.dry_run:
            receipt["status"] = "validated_dry_run"
            save_json(output / "request-preview.json", dict(
                request, vocal_audio_base64_bytes=4 * ((len(input_bytes) + 2) // 3)
            ))
        else:
            key = os.getenv("STEP_API_KEY", "").strip()
            if not key:
                receipt["status"] = "blocked_no_credential"
                return 2
            request["vocal_audio"] = base64.b64encode(input_bytes).decode("ascii")
            save_json(output / "request.json", request)
            receipt["status"] = "submission_started"
            save_json(output / "receipt.json", receipt)
            run_remote(request, key, args.timeout_seconds, output, receipt)
    except (StepTaskFailed, StepHTTPError, URLError, TimeoutError, KeyError, ValueError,
            subprocess.CalledProcessError) as exc:
        receipt["status"] = "timeout" if isinstance(exc, TimeoutError) else "failed"
        receipt["error_type"] = type(exc).__name__
        if isinstance(exc, StepHTTPError):
            receipt["http_status"] = exc.status_code
            receipt["service_error_type"] = exc.error_type
        return 1
    finally:
        receipt["total_seconds"] = time.monotonic() - started
        receipt["ended_at_awst"] = awst_now()
        save_json(output / "receipt.json", receipt)
        print(receipt["status"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
