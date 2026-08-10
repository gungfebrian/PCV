"""
Fine-tune MegaDescriptor-T dengan ArcFace subcenter di SeaTurtleIDHeads.

Resep dari riset (Notion bagian 20): semua sistem teratas — paper
SeaTurtleID2022 (86.8%) dan pemenang Turtle Recall (97.6%) — memakai metric
learning ArcFace yang di-fine-tune pada wajah penyu, bukan embedding generik.
ArcFace melatih eksplisit "individu sama merapat, individu beda menjauh",
persis obat untuk tumpang-tindih distribusi yang terukur di bagian 17.

SPLIT TIME-AWARE SEJAK AWAL (bagian 20): foto TERAKHIR secara kronologis per
individu dipisahkan untuk evaluasi; kalau individu punya foto lintas tahun,
seluruh tahun terakhirnya jadi data uji. Melatih dan menguji pada hari
pemotretan yang sama = angka bohong.

Augmentasi TANPA horizontal flip — sisi kiri dan kanan kepala penyu adalah
pola yang berbeda (bagian 22: pisah sisi = +17-20 poin). Flip akan
mengajarkan model bahwa kiri == kanan, kebalikan dari yang kita mau.

Jalankan:
    .venv/bin/python latih_arcface.py            # latih + evaluasi
    .venv/bin/python latih_arcface.py --eval     # evaluasi bobot tersimpan saja

Keluaran: model_arcface.pt (dipakai tombol 'ArcFace' di visualizer).
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE_DIR, "dataset_penyu", "SeaTurtleIDHeads")
BOBOT = os.path.join(BASE_DIR, "model_arcface.pt")

EPOCHS = 8
BATCH = 32
IMG = 224
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3


# ------------------------------------------------------------------- data
def muat_daftar():
    """Return train_list, eval_gal, eval_uji sebagai (path, label_str)."""
    with open(os.path.join(DATA, "annotations.json")) as f:
        anot = json.load(f)

    byind = {}
    for im in anot["images"]:
        ind = im["path"].split("/")[1]
        try:
            d = datetime.strptime(im["date"].split()[0], "%Y:%m:%d")
        except (ValueError, KeyError):
            continue
        p = os.path.join(DATA, im["path"])
        if os.path.exists(p):
            byind.setdefault(ind, []).append((d, p))

    train, eval_gal, eval_uji = [], {}, []
    for ind, lst in byind.items():
        lst.sort()
        tahun = sorted({d.year for d, _ in lst})
        if len(tahun) >= 2 and len(lst) >= 8:
            # Individu multi-tahun: tahun terakhir DISIMPAN untuk evaluasi,
            # model tidak pernah melihatnya saat latihan.
            th_uji = tahun[-1]
            latih = [(p, ind) for d, p in lst if d.year != th_uji]
            uji = [(p, ind) for d, p in lst if d.year == th_uji]
            train += latih
            # galeri evaluasi = 4 foto awal dari data latih (bukan data uji)
            eval_gal[ind] = [p for p, _ in latih[:4]]
            eval_uji += uji[:10]
        else:
            train += [(p, ind) for _, p in lst]

    return train, eval_gal, eval_uji


# ------------------------------------------------------------------ latih
def latih():
    import cv2
    import timm
    import torch
    from pytorch_metric_learning.losses import SubCenterArcFaceLoss
    from torch.utils.data import DataLoader, Dataset

    train, eval_gal, eval_uji = muat_daftar()
    label_set = sorted({l for _, l in train})
    lab2id = {l: i for i, l in enumerate(label_set)}
    print(f"latih : {len(train)} foto, {len(label_set)} individu")
    print(f"eval  : {len(eval_gal)} individu galeri, {len(eval_uji)} foto uji "
          f"(tahun yang tidak pernah dilihat model)")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    class DS(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            p, lab = self.items[i]
            im = cv2.imread(p)
            if im is None:
                im = np.zeros((IMG, IMG, 3), np.uint8)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            # Augmentasi ringan: crop acak sedikit + jitter warna.
            # TANPA flip horizontal — lihat catatan modul.
            h, w = im.shape[:2]
            if h > 32 and w > 32:
                m = np.random.randint(0, max(1, int(0.08 * min(h, w))) + 1, 4)
                im = im[m[0]:h - m[1] or h, m[2]:w - m[3] or w]
            im = cv2.resize(im, (IMG, IMG), interpolation=cv2.INTER_AREA)
            im = im.astype(np.float32) / 255.0
            im *= np.random.uniform(0.85, 1.15)          # brightness jitter
            im = np.clip(im, 0, 1)
            im = (im - MEAN) / STD
            return torch.from_numpy(im.transpose(2, 0, 1)), lab2id[lab]

    dl = DataLoader(DS(train), batch_size=BATCH, shuffle=True,
                    num_workers=0, drop_last=True)

    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224",
                              pretrained=True, num_classes=0).to(device)
    # Subcenter (K=3): tiap individu punya 3 pusat — menampung variasi pose
    # tanpa memaksa semua fotonya ke satu titik. Ini pembeda resep pemenang.
    loss_fn = SubCenterArcFaceLoss(num_classes=len(label_set),
                                   embedding_size=768, sub_centers=3).to(device)
    opt = torch.optim.AdamW([
        {"params": model.parameters(), "lr": LR_BACKBONE},
        {"params": loss_fn.parameters(), "lr": LR_HEAD},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS * len(dl))

    model.train()
    t0 = time.time()
    for ep in range(EPOCHS):
        total = jml = 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            emb = model(x)
            loss = loss_fn(emb, y)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss)
            jml += 1
        print(f"epoch {ep+1}/{EPOCHS}  loss {total/max(1,jml):.4f}  "
              f"({(time.time()-t0)/60:.1f} mnt)", flush=True)
        torch.save(model.state_dict(), BOBOT)      # simpan tiap epoch

    print(f"selesai latih {(time.time()-t0)/60:.1f} menit -> {BOBOT}")
    return evaluasi()


# ------------------------------------------------------------------- eval
def evaluasi():
    """Evaluasi time-aware: galeri dari tahun latihan, uji dari tahun
    yang tidak pernah dilihat model. Protokol identik dengan bagian 20."""
    import cv2
    import timm
    import torch

    _, eval_gal, eval_uji = muat_daftar()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224",
                              pretrained=True, num_classes=0)
    # weights_only=True: file bobot hanya berisi tensor; tanpa ini torch.load
    # meng-unpickle objek Python sembarang — file yang dirusak bisa
    # mengeksekusi kode saat dimuat.
    model.load_state_dict(torch.load(BOBOT, map_location="cpu",
                                     weights_only=True))
    model = model.to(device).eval()

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def emb(path):
        im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (IMG, IMG), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(im).float().div_(255).permute(2, 0, 1)
        x = ((x - MEAN) / STD).unsqueeze(0).to(device)
        with torch.no_grad():
            v = model(x)[0].float().cpu().numpy()
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    gal = {k: [emb(p) for p in ps] for k, ps in eval_gal.items()}
    b = t5 = n = 0
    sama, beda = [], []
    for p, truth in eval_uji:
        v = emb(p)
        rk = sorted(((k, (1 - max(float(np.dot(v, w)) for w in vs)) / 2)
                     for k, vs in gal.items()), key=lambda x: x[1])
        b += rk[0][0] == truth
        t5 += truth in [r[0] for r in rk[:5]]
        n += 1
        for k, d in rk:
            (sama if k == truth else beda).append(d)

    sama, beda = np.array(sama), np.array(beda)
    print(f"\nARCFACE time-aware  ({len(gal)} individu, {n} foto uji)")
    print(f"  Top-1 {b/n*100:5.1f}%   Top-5 {t5/n*100:5.1f}%")
    print(f"  jarak SAMA {sama.mean():.4f} +/- {sama.std():.4f}")
    print(f"  jarak BEDA {beda.mean():.4f} +/- {beda.std():.4f}")
    print(f"\npembanding MegaDescriptor-T generik, protokol sama: Top-1 60.6%")
    print(f"\nSTATS untuk turtle_mode.py:")
    print(f'  "arcface": {{"sama": ({sama.mean():.4f}, {sama.std():.4f}), '
          f'"beda": ({beda.mean():.4f}, {beda.std():.4f})}}')
    return b / n


if __name__ == "__main__":
    if "--eval" in sys.argv:
        evaluasi()
    else:
        latih()
