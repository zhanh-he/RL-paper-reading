"""Provide glibc >=2.31 for the frozen vLLM wheel without editing its locks."""

import argparse
import subprocess
from pathlib import Path

from generate_objective import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--python', required=True)
    a = p.parse_args()
    binary = '/opt/singularity/bin/singularity'
    image = a.work / 'python312-bookworm-build.sif'
    uri = 'docker://python:3.12.11-bookworm'
    if not image.exists():
        subprocess.run([binary, 'pull', '--disable-cache', str(image), uri], check=True)
    # The native /apps Python needs these RHEL compatibility libraries, but
    # libc/loader and the metrics interpreter come from Debian/portable UV.
    binds = ['/apps', '/g/data/wa66', '/scratch/wa66', '/scratch/iv96', '/jobfs']
    for name in ['libffi.so.6', 'libssl.so.1.1', 'libcrypto.so.1.1']:
        binds.append(f'/usr/lib64/{name}:/usr/lib/x86_64-linux-gnu/{name}')
    probe = subprocess.check_output([binary, 'exec', '--bind', ','.join(binds), str(image), a.python,
        '-c', 'import ssl,ctypes,torch,yue2,muq; import platform; print(platform.libc_ver(),torch.__version__)'],
        text=True)
    compiler = subprocess.check_output([binary, 'exec', str(image), 'cc', '--version'], text=True)
    write(a.work / 'container-runtime.json', {'image': str(image), 'image_sha256': sha(image),
        'source': uri, 'binary': binary, 'binds': binds, 'preflight': probe.strip(),
        'compiler_preflight': compiler.splitlines()[0],
        'scope': 'OS compatibility wrapper; native generator and frozen scorer dependencies unchanged'})
    print(probe, flush=True)


if __name__ == '__main__':
    main()
