from pathlib import Path
import shutil
import sys

BASE_DIR = Path("/sdf/group/rp/rajarshipc")


def copy_study_files(index: int) -> None:
    source_dir = BASE_DIR / f"Pen_study_New_{index}"
    destination_dir = BASE_DIR / f"Copy_{index}"

    if not source_dir.is_dir():
        print(f"[WARNING] Source directory does not exist: {source_dir}")
        return

    destination_dir.mkdir(parents=True, exist_ok=True)

    # Copy lcls2CP.inp into Copy_i/
    input_file = source_dir / "lcls2CP.inp"
    if input_file.is_file():
        shutil.copy2(input_file, destination_dir / input_file.name)
    else:
        print(f"[WARNING] File not found: {input_file}")

    # Copy everything under Pen_study_New_i/analysis/ into Copy_i/analysis/
    source_analysis = source_dir / "analysis"
    destination_analysis = destination_dir / "analysis"

    if source_analysis.is_dir():
        shutil.copytree(
            source_analysis,
            destination_analysis,
            dirs_exist_ok=True,
            copy_function=shutil.copy2,
        )
    else:
        print(f"[WARNING] Analysis directory not found: {source_analysis}")

    print(f"[OK] Processed {source_dir} -> {destination_dir}")


def main() -> int:
    for index in range(1, 101):
        try:
            copy_study_files(index)
        except PermissionError as error:
            print(f"[ERROR] Permission denied for index {index}: {error}")
        except OSError as error:
            print(f"[ERROR] Could not process index {index}: {error}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
