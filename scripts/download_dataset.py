"""Download the LLVIP archive and verify its published MD5 checksum."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

import gdown


LLVIP_DATASET_URL = "https://drive.google.com/uc?id=1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv"
EXPECTED_MD5 = "e64affb4b0b50e1772ff6f67da873bf6"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - required to verify the published file
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("LLVIP.zip"))
    parser.add_argument("--force", action="store_true", help="download even if valid")
    args = parser.parse_args()

    if args.output.is_file() and not args.force:
        actual_md5 = file_md5(args.output)
        if actual_md5 == EXPECTED_MD5:
            print(f"Verified existing {args.output} ({actual_md5})")
            return
        print(f"Existing archive has MD5 {actual_md5}; downloading a replacement")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=args.output.parent,
        prefix=f".{args.output.name}-",
        suffix=".download",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        result = gdown.download(
            url=LLVIP_DATASET_URL,
            output=str(temporary_path),
            quiet=False,
        )
        if result is None or not temporary_path.is_file():
            raise RuntimeError("gdown did not produce an archive")
        actual_md5 = file_md5(temporary_path)
        if actual_md5 != EXPECTED_MD5:
            raise ValueError(
                f"downloaded archive MD5 mismatch: expected {EXPECTED_MD5}, "
                f"got {actual_md5}"
            )
        temporary_path.replace(args.output)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"Downloaded and verified {args.output} ({EXPECTED_MD5})")


if __name__ == "__main__":
    main()
