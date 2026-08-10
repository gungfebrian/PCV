"""
MegaDescriptor asli — deskriptor terlatih untuk re-identification satwa liar.

Dilatih di puluhan dataset satwa (termasuk TurtleID2022 dari wildlife-datasets),
jadi embedding-nya sudah tahu bahwa "individu yang sama dari sudut berbeda"
harus berdekatan — hal yang mustahil dipelajari deskriptor piksel.

Dipakai lewat antarmuka yang sama dengan turtle_mode.deskriptor_baseline:

    vektor = deskriptor(bgr)   -> np.ndarray 1-D, sudah L2-normalized

Model dimuat malas dan hanya sekali. Unduhan pertama beberapa ratus MB.

Varian:
    T-224   Swin-Tiny,  27.5 juta parameter, 768-dim   — cepat, cocok realtime
    L-384   Swin-Large, 197 juta parameter, 1536-dim   — paling akurat, berat
"""

import numpy as np

VARIAN = {
    "T": ("hf-hub:BVRA/MegaDescriptor-T-224", 224),
    "L": ("hf-hub:BVRA/MegaDescriptor-L-384", 384),
    # Backbone T yang di-fine-tune ArcFace di SeaTurtleIDHeads
    # (latih_arcface.py). Bobotnya lokal, bukan unduhan.
    "arcface": ("hf-hub:BVRA/MegaDescriptor-T-224", 224),
}

import os as _os
BOBOT_ARCFACE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                              "model_arcface.pt")


def arcface_tersedia():
    return _os.path.exists(BOBOT_ARCFACE)

_model = {}      # varian -> (model, ukuran, device)


def tersedia():
    """Cek torch+timm terpasang tanpa benar-benar memuat model."""
    try:
        import timm  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def muat(varian="T"):
    """Muat model sekali dan simpan di cache proses."""
    if varian in _model:
        return _model[varian]

    import timm
    import torch

    nama, ukuran = VARIAN[varian]
    model = timm.create_model(nama, pretrained=True, num_classes=0)
    if varian == "arcface":
        if not arcface_tersedia():
            raise FileNotFoundError(
                "model_arcface.pt belum ada. Latih dulu: "
                ".venv/bin/python latih_arcface.py")
        # weights_only=True: hanya tensor yang boleh dimuat — tanpa ini,
        # file bobot yang dirusak bisa mengeksekusi kode saat di-unpickle.
        model.load_state_dict(torch.load(BOBOT_ARCFACE, map_location="cpu",
                                         weights_only=True))
    model.eval()

    # MPS = GPU Apple Silicon. Jauh lebih cepat daripada CPU untuk Swin.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    _model[varian] = (model, ukuran, device)
    return _model[varian]


def deskriptor(bgr, varian="T"):
    """BGR (OpenCV) -> embedding L2-normalized."""
    import cv2
    import torch

    model, ukuran, device = muat(varian)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (ukuran, ukuran), interpolation=cv2.INTER_AREA)

    # Normalisasi ImageNet — model dilatih dengan statistik ini, memakai yang
    # lain akan menggeser seluruh distribusi embedding.
    x = torch.from_numpy(rgb).float().div_(255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = ((x - mean) / std).unsqueeze(0).to(device)

    with torch.no_grad():
        v = model(x)[0].float().cpu().numpy()

    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def deskriptor_batch(list_bgr, varian="T", batch=16):
    """Versi batch untuk membangun galeri — jauh lebih cepat dari satu per satu."""
    import cv2
    import torch

    model, ukuran, device = muat(varian)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    keluar = []
    for i in range(0, len(list_bgr), batch):
        potong = list_bgr[i:i + batch]
        xs = []
        for bgr in potong:
            rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                             (ukuran, ukuran), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(rgb).float().div_(255.0).permute(2, 0, 1)
            xs.append((t - mean) / std)
        x = torch.stack(xs).to(device)
        with torch.no_grad():
            v = model(x).float().cpu().numpy()
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        keluar.append(v)
    return np.concatenate(keluar) if keluar else np.zeros((0, 768), np.float32)
