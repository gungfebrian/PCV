"""
Jalankan satu kondisi preprocessing dan simpan embedding-nya.

Sengaja resumable dan satu-kondisi-per-panggilan: lingkungan eksekusi di sini
punya batas waktu per perintah, jadi pekerjaan dipecah dan dicatat ke disk.
Efek sampingnya bagus — kalau satu kondisi gagal, yang lain tidak ikut hilang.

    python3 jalankan.py raw
    python3 jalankan.py --semua
    python3 jalankan.py --status
"""

import json
import os
import sys
import time

import numpy as np

import protokol as P

HASIL = P.dir_hasil()
os.makedirs(HASIL, exist_ok=True)


def siapkan():
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    return kat, gal, qry


def tulis_header(kat, gal, qry, cfg):
    import timm
    import torch
    meta = {
        "dataset": "SeaTurtleIDHeads",
        "dataset_hash": P.hash_dataset(kat),
        "model": {"T": "BVRA/MegaDescriptor-T-224",
                  "L": "BVRA/MegaDescriptor-L-384"}[P.MODEL],
        "model_arch": cfg["architecture"],
        "model_snapshot": os.path.basename(P.SNAP_T or ""),
        "input_size": P.UKURAN, "crop_pct": P.CROP_PCT,
        "torch": torch.__version__, "timm": timm.__version__,
        "numpy": np.__version__,
        "sanity": P.periksa_split(gal, qry),
    }
    with open(os.path.join(HASIL, "header.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def jalankan_kondisi(nama, gal, qry, model, budget=None):
    """Hitung embedding gallery+query untuk satu kondisi. Resumable per shard."""
    paths = [r["path"] for r in gal] + [r["path"] for r in qry]
    out = os.path.join(HASIL, f"emb_{nama}.npy")
    prog = os.path.join(HASIL, f"emb_{nama}.progress")
    done = int(open(prog).read()) if os.path.exists(prog) else 0
    E = np.load(out) if os.path.exists(out) else np.zeros((len(paths), P.DIM), np.float32)
    t0 = time.time()
    # L-384 ~15x lebih lambat dari T-224 di CPU; shard harus lebih kecil supaya
    # progres tetap tersimpan sebelum batas waktu perintah tercapai.
    STEP = 16 if P.MODEL == "L" else 64
    while done < len(paths):
        blok = paths[done:done + STEP]
        E[done:done + len(blok)] = P.embed(blok, nama, model)
        done += len(blok)
        np.save(out, E)
        open(prog, "w").write(str(done))
        if budget and time.time() - t0 > budget:
            break
    return done, len(paths)


def main():
    args = sys.argv[1:]
    kat, gal, qry = siapkan()

    if "--status" in args:
        print(json.dumps(P.periksa_split(gal, qry), indent=2))
        for k in P.KONDISI:
            p = os.path.join(HASIL, f"emb_{k}.progress")
            d = int(open(p).read()) if os.path.exists(p) else 0
            print(f"  {k:9} {d}/{len(gal)+len(qry)}")
        return

    budget = 35.0
    for a in args:
        if a.startswith("--budget="):
            budget = float(a.split("=")[1])

    model, cfg = P.muat_model()
    tulis_header(kat, gal, qry, cfg)

    semua = list(P.KONDISI) + list(P.KONDISI_BERKAS)
    kondisi = [a for a in args if a in semua] or list(P.KONDISI)
    for k in kondisi:
        d, n = jalankan_kondisi(k, gal, qry, model, budget=budget)
        print(f"{k:9} {d}/{n}")
        if d < n:
            break


if __name__ == "__main__":
    main()
