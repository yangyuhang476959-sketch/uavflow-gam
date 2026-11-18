import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import zstandard


def test_extract_checksum_is_opt_in(tmp_path):
    root = tmp_path / 'source'
    (root / 'data').mkdir(parents=True)
    (root / 'metadata').mkdir()
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode='w') as archive:
        data = b'test depth shard'
        entry = tarfile.TarInfo('depth/example.txt')
        entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))
    (root / 'data' / 'part.tar.zst').write_bytes(zstandard.ZstdCompressor().compress(tar_bytes.getvalue()))
    (root / 'metadata' / 'shards.json').write_text(json.dumps({'episodes': 1, 'shards': [{'name': 'part.tar.zst', 'sha256': 'wrong'}]}))
    for name in ('episodes.csv', 'source_summary.json', 'instruction_overrides.json'):
        (root / 'metadata' / name).write_text('{}')
    script = Path(__file__).resolve().parents[1] / 'scripts/extract_uavflow_depth.py'
    command = [sys.executable, str(script), '--dataset-root', str(root), '--output', str(tmp_path / 'out')]
    subprocess.run(command, check=True)
    assert (tmp_path / 'out/depth/example.txt').read_bytes() == data
    checked = subprocess.run(command + ['--verify-checksums'], capture_output=True, text=True)
    assert checked.returncode != 0
    assert 'Checksum mismatch' in checked.stderr
