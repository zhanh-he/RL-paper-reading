"""Finite asset preparation. No GPU use, scheduled tasks, or training claims."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi

RELEASES = {
    'diffrhythm2': [('ASLP-lab/DiffRhythm2', '9aa15742e4889c0eb2e198db6fdab1facf1b6761', '.')],
    'heartmula': [('HeartMuLa/HeartMuLaGen', '2e18e01702011f4f7dc8642b6260967df62417ea', '.'),
                  ('HeartMuLa/HeartMuLa-oss-3B-happy-new-year', '41f6fc68490e11dc43fdabaa6b5767946408c903', 'HeartMuLa-oss-3B'),
                  ('HeartMuLa/HeartCodec-oss-20260123', 'f889dab0532cfa4bf459f2a3367eb6d346b8eeda', 'HeartCodec-oss')],
    'songbloom': [('rsxdalv/SongBloom', '093351709a0efa263569cc4b679429d38f4508ec', '.')],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', choices=RELEASES, required=True)
    a = p.parse_args(); root = a.root/a.model; root.mkdir(parents=True, exist_ok=True)
    receipts = []
    for repo, revision, subdir in RELEASES[a.model]:
        allow = None if a.model != 'songbloom' else ['README.md', 'songbloom_full_150s.pt', 'songbloom_full_150s.yaml',
                 'autoencoder_music_dsp1920.ckpt', 'stable_audio_1920_vae.json', 'vocab_g2p.yaml']
        snapshot_download(repo, revision=revision, local_dir=root/subdir, allow_patterns=allow,
                          cache_dir=a.root/'hf-cache', max_workers=4)
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        for item in info.siblings:
            path = root/subdir/item.rfilename
            if not path.is_file() or '.cache' in path.parts: continue
            h = hashlib.sha256()
            with path.open('rb') as stream:
                while block := stream.read(8 << 20): h.update(block)
            digest = h.hexdigest()
            if item.lfs and digest != item.lfs.sha256: raise ValueError(f'LFS hash mismatch: {path.name}')
            receipts.append({'repo': repo, 'revision': revision, 'file': str(path.relative_to(root)), 'sha256': digest})
    report = {'model': a.model, 'files': receipts, 'download_complete': True,
              'source_status': 'community-restored, not independently matched to unavailable author endpoint' if a.model == 'songbloom' else 'author release',
              'training_started': False}
    (root/'asset-receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'model': a.model, 'files': len(receipts), 'download_complete': True}), flush=True)


if __name__ == '__main__':
    main()
