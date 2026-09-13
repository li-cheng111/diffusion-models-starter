from pathlib import Path

from projects.project3_stable_diffusion.experiment_utils import add_file_hashes, sha256_file, write_json


def test_hash_and_json_roundtrip(tmp_path: Path):
    data = tmp_path / "sample.txt"
    data.write_text("stable diffusion", encoding="utf-8")
    target = tmp_path / "meta.json"
    metadata = add_file_hashes({"ok": True}, [data])
    write_json(target, metadata)
    assert len(metadata["output_sha256"][str(data)]) == 64
    assert target.exists()
    assert sha256_file(data) == metadata["output_sha256"][str(data)]
