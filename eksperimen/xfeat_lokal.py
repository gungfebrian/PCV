"""
XFeat (CVPR 2024) — arsitektur direkonstruksi dari state_dict.

Kenapa direkonstruksi, bukan diimpor: `bobot_matcher/xfeat.pt` hanya berisi
state_dict, sedangkan kelas modelnya ada di repo GitHub yang tidak bisa
dijangkau dari lingkungan eksekusi ini. Bentuk tiap tensor di state_dict
menentukan arsitekturnya secara unik, jadi ia bisa dibangun ulang — dan
**dibuktikan benar** lewat dua pemeriksaan:

  1. `load_state_dict(strict=True)` harus lolos tanpa satu pun key hilang
     atau berlebih. Kalau arsitekturnya meleset satu lapis saja, ini gagal.
  2. Mencocokkan sebuah gambar dengan DIRINYA SENDIRI harus menghasilkan
     ratusan match dengan inlier hampir sempurna. Bobot yang termuat di
     tempat yang salah tidak akan lolos uji ini.

Keduanya dijalankan oleh `uji.py`. Tanpa keduanya, angka XFeat di laporan
tidak boleh dipercaya.

Catatan penting: BatchNorm di XFeat memakai `affine=False` — terlihat dari
state_dict yang punya `running_mean`/`running_var` tapi TIDAK punya
`weight`/`bias` untuk lapisan norm. Memakai affine=True akan membuat
strict-load gagal, dan itu memang yang diinginkan.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class BasicLayer(nn.Module):
    def __init__(self, in_c, out_c, k=3, stride=1, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_c, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class XFeatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)
        self.skip1 = nn.Sequential(
            nn.AvgPool2d(4, stride=4),
            nn.Conv2d(1, 24, 1, stride=1, padding=0))
        self.block1 = nn.Sequential(
            BasicLayer(1, 4, stride=1), BasicLayer(4, 8, stride=2),
            BasicLayer(8, 8, stride=1), BasicLayer(8, 24, stride=2))
        self.block2 = nn.Sequential(
            BasicLayer(24, 24, stride=1), BasicLayer(24, 24, stride=1))
        self.block3 = nn.Sequential(
            BasicLayer(24, 64, stride=2), BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, 1, padding=0))
        self.block4 = nn.Sequential(
            BasicLayer(64, 64, stride=2), BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1))
        self.block5 = nn.Sequential(
            BasicLayer(64, 128, stride=2), BasicLayer(128, 128, stride=1),
            BasicLayer(128, 128, stride=1), BasicLayer(128, 64, 1, padding=0))
        self.block_fusion = nn.Sequential(
            BasicLayer(64, 64, stride=1), BasicLayer(64, 64, stride=1),
            nn.Conv2d(64, 64, 1, padding=0))
        self.heatmap_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0), BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 1, 1), nn.Sigmoid())
        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0), BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0), nn.Conv2d(64, 65, 1))
        self.fine_matcher = nn.Sequential(
            nn.Linear(128, 512), nn.BatchNorm1d(512, affine=False), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.BatchNorm1d(512, affine=False), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.BatchNorm1d(512, affine=False), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.BatchNorm1d(512, affine=False), nn.ReLU(inplace=True),
            nn.Linear(512, 64))

    @staticmethod
    def _unfold2d(x, ws=2):
        B, C, H, W = x.shape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws).reshape(
            B, C, H // ws, W // ws, ws ** 2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H // ws, W // ws)

    def forward(self, x):
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=True)
            x = self.norm(x)
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x4 = Fn.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode="bilinear")
        x5 = Fn.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode="bilinear")
        feats = self.block_fusion(x3 + x4 + x5)
        heatmap = self.heatmap_head(feats)
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8))
        return feats, keypoints, heatmap


BOBOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "bobot_matcher", "xfeat.pt")


def muat(path=BOBOT):
    """Muat bobot dengan strict=True. Sengaja tidak memakai strict=False:
    kalau arsitekturnya meleset, kita ingin tahu sekarang, bukan lewat angka
    yang pelan-pelan salah."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} tidak ada — jalankan `unduh_matcher.py` di Mac dulu")
    m = XFeatModel()
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m.load_state_dict(sd, strict=True)      # akan melempar kalau tidak cocok
    m.eval()
    return m


# ------------------------------------------------------- deteksi + deskripsi
def _nms(x, ambang=0.05, ks=5):
    pad = ks // 2
    lokal_maks = nn.MaxPool2d(ks, stride=1, padding=pad)(x)
    pos = (x == lokal_maks) & (x > ambang)
    return pos


@torch.no_grad()
def ekstrak(model, gray_u8, top_k=2048, sisi=800):
    """Gambar grayscale uint8 -> (koordinat Nx2, deskriptor NxD L2-normalized).

    Bentuk keluarannya sengaja dibuat sama persis dengan `rerank.Klasik`
    supaya fungsi skor yang sama bisa dipakai untuk keduanya — kalau tidak,
    perbandingan SIFT vs XFeat akan membandingkan dua definisi skor berbeda.
    """
    import cv2
    im = gray_u8
    s = sisi / max(im.shape)
    if s < 1:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    else:
        s = 1.0
    H, W = im.shape
    H8, W8 = (H // 32) * 32, (W // 32) * 32
    if H8 < 32 or W8 < 32:
        return None
    im = im[:H8, :W8]
    x = torch.from_numpy(im).float()[None, None] / 255.0

    feats, kpts_logits, heatmap = model(x)
    feats = Fn.normalize(feats, dim=1)

    # kepala keypoint: 65 kanal = 8x8 sel + 1 "tidak ada keypoint"
    # 65 kanal = 8x8 sel + satu kanal "tidak ada keypoint" yang dibuang.
    # Menyusun 64 kanal itu kembali jadi peta resolusi penuh persis pekerjaan
    # pixel_shuffle; menulisnya manual dengan permute/reshape mudah salah dan
    # memang sempat salah di sini.
    prob = kpts_logits.softmax(dim=1)[:, :64]
    prob = Fn.pixel_shuffle(prob, 8)                       # (1,1,Hc*8,Wc*8)
    skor = prob * Fn.interpolate(heatmap, size=prob.shape[-2:], mode="bilinear")

    pos = _nms(skor, ambang=0.02, ks=5)[0, 0]
    ys, xs = torch.nonzero(pos, as_tuple=True)
    if len(xs) == 0:
        return None
    nilai = skor[0, 0][ys, xs]
    if len(xs) > top_k:
        pilih = torch.topk(nilai, top_k).indices
        ys, xs = ys[pilih], xs[pilih]

    # ambil deskriptor pada posisi keypoint (grid 1/8 resolusi)
    gx = (xs.float() / (W8 - 1) * 2 - 1)
    gy = (ys.float() / (H8 - 1) * 2 - 1)
    grid = torch.stack([gx, gy], dim=-1)[None, None]
    des = Fn.grid_sample(feats, grid, align_corners=True, mode="bilinear")
    des = Fn.normalize(des[0, :, 0].t(), dim=1)

    pts = torch.stack([xs.float(), ys.float()], dim=1).numpy() / s
    return pts.astype(np.float32), des.numpy().astype(np.float32)


def cocokkan(a, b, min_cossim=0.82):
    """Mutual nearest neighbour + ambang cosine — matcher bawaan XFeat.

    BUKAN ratio test Lowe. Deskriptor XFeat 64-dim sangat padat, sehingga
    tetangga terdekat kedua sering hampir sedekat yang pertama; ratio test
    di situ membuang hampir semua pasangan. Diuji: dengan ratio 0.8 hanya
    2 pasangan yang lolos antar gambar berbeda, padahal self-match sempurna.

    Memaksa XFeat lewat ratio test akan membuatnya terlihat jauh lebih buruk
    dari SIFT karena alasan yang tidak ada hubungannya dengan kualitas
    deskriptornya. Tiap metode dipakai dengan matcher yang memang dirancang
    untuknya; yang disamakan adalah skor akhirnya — jumlah inlier RANSAC.
    """
    if a is None or b is None:
        return None, None
    da, db = torch.from_numpy(a[1]), torch.from_numpy(b[1])
    cos = da @ db.t()
    m12 = cos.argmax(dim=1)
    m21 = cos.argmax(dim=0)
    idx = torch.arange(len(m12))
    mutual = m21[m12] == idx
    if min_cossim > 0:
        mutual &= cos.max(dim=1).values > min_cossim
    i = idx[mutual].numpy()
    j = m12[mutual].numpy()
    return a[0][i], b[0][j]
