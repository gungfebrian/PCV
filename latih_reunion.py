"""Peluncur untuk alur detektor kepala YOLO Reunion.

Jalankan dari root repo PCV:
    python latih_reunion.py --prepare-annotation

Skrip aslinya tinggal di worktree `turtle-reid-prototype`, yang di-ignore
git. Jadi di clone yang bersih file itu TIDAK ADA — makanya keberadaannya
diperiksa dulu supaya pesan errornya jelas, bukan FileNotFoundError dari
dalam runpy.
"""

import runpy
import sys
from pathlib import Path

SKRIP = (Path(__file__).resolve().parent
         / ".worktrees" / "turtle-reid-prototype"
         / "eksperimen" / "latih_detektor_reunion.py")


def main():
    if not SKRIP.is_file():
        sys.exit(
            f"Skrip detektor tidak ditemukan:\n  {SKRIP}\n\n"
            "Worktree `turtle-reid-prototype` belum ada di mesin ini "
            "(folder .worktrees/ di-ignore git). Buat dulu:\n"
            "  git worktree add .worktrees/turtle-reid-prototype "
            "codex/turtle-reid-prototype"
        )
    runpy.run_path(str(SKRIP), run_name="__main__")


if __name__ == "__main__":
    main()
