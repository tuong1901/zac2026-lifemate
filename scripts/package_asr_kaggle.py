"""Package only src/asr code for a private Kaggle code Dataset (no .env/data)."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent.parent


def package(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for file in sorted((ROOT / "src/asr").glob("*.py")):
            archive.write(file, file.relative_to(ROOT).as_posix())
        for name in ("src/__init__.py", "docs/KAGGLE_PHOWHISPER.md"):
            file = ROOT / name
            if file.is_file():
                archive.write(file, name)
    print(f"Kaggle code package: {destination.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/kaggle_asr/lifemate_asr_code.zip")
    package(parser.parse_args().out)
