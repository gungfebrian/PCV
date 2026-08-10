# PCV — Eksperimen Preprocessing Re-ID Penyu

Eksperimen, bukan produk. Satu pertanyaan yang dijawab: **preprocessing mana
yang benar-benar menaikkan akurasi re-identifikasi penyu, dan seberapa besar?**

## Jawabannya: tidak ada

Kelima kondisi preprocessing yang diuji **tidak bisa dibedakan dari raw** —
semua p-value di 0.26–0.59, semua selang kepercayaan melewati nol.

| Kondisi | Rank-1 | Rank-5 | mAP | Δ Rank-1 vs raw (95% CI) | p |
|---|---|---|---|---|---|
| **Raw (baseline)** | **25.00** | 46.43 | **37.40** | — | — |
| Crop kepala (center 70%) | 22.62 | 46.43 | 34.49 | −2.38 [−8.33, +3.57] | 0.557 |
| White balance (gray-world) | 22.62 | 48.21 | 35.59 | −2.38 [−7.74, +2.98] | 0.503 |
| CLAHE (L, clip 2.0) | 21.43 | 47.62 | 34.71 | −3.57 [−9.52, +2.38] | 0.327 |
| Grayscale | 20.24 | 46.43 | 32.73 | −4.76 [−11.90, +2.38] | 0.256 |
| Crop + WB | 22.62 | **51.19** | 35.18 | −2.38 [−8.93, +4.17] | 0.585 |

ReunionTurtles · 84 individu (50 hijau + 34 sisik) · MegaDescriptor-L-384
frozen · n = 168 query.

Yang **justru** menaikkan akurasi ada di luar preprocessing: ganti T-224 ke
L-384 (18.45 → 25.00), dan konsensus dua sisi (25.00 → 30.95, Δ mAP signifikan
tapi butuh dua foto per penyu).

Laporan lengkap: **[`eksperimen/README.md`](eksperimen/README.md)** ·
audit repo: **[`eksperimen/AUDIT.md`](eksperimen/AUDIT.md)** ·
progres per checkpoint ada di Notion (Project → Research for Deep learning).

## Struktur

```
PCV/
├── eksperimen/          <- eksperimen: protokol §3 dikunci di sini
│   ├── protokol.py          protokol §3 dikunci: split, matching, metrik
│   ├── jalankan.py          hitung embedding per kondisi (resumable)
│   ├── evaluasi.py          metrik + breakdown + validasi wildlife-tools
│   ├── statistik.py         McNemar + bootstrap CI
│   ├── transform_ab.py      A/B transform input
│   ├── lanjutan.py          konsensus / hubness / fusi T+L
│   ├── sanity.py            sanity check §8
│   ├── kasus_gagal.py       5 kasus gagal + top-5 gallery
│   ├── app_demo.py          demo Streamlit (5 tab)
│   ├── grafik_notion.py     SVG ringkas untuk laporan
│   ├── uji.py               22 tes invarian protokol
│   ├── hasil/               embedding + metrik per run
│   ├── AUDIT.md
│   └── README.md
├── aplikasi/            <- re-ID penyu realtime dari kamera (di luar scope §1)
│   ├── penyu_live.py        loop kamera + pipeline lengkap + stage-2
│   ├── tampilan.py          komposisi kanvas (fungsi murni, bisa diuji)
│   └── uji_tampilan.py      18 tes tata letak, tanpa layar
├── dataset_penyu/
│   ├── ReunionTurtles/      dataset utama (336 foto, 84 individu)
│   ├── SeaTurtleIDHeads/    pembanding skala (7.582 foto)
│   └── ZindiTurtleRecall/   tidak dipakai
└── tidak_penting/       <- proyek lain + kode lama, lihat README di dalamnya
```

## Menjalankan

Semua paket ada di `.venv/`, **bukan** di Python global. Tidak ada `pip`
telanjang di PATH — pakai `source .venv/bin/activate` atau awali dengan
`.venv/bin/`.

```bash
source .venv/bin/activate
cd eksperimen

MODEL=L python3 uji.py           # 22 tes invarian — jalankan ini dulu
python3 jalankan.py --status     # cek split & progres
MODEL=L python3 jalankan.py      # semua kondisi (resumable, panggil ulang)
MODEL=L python3 statistik.py     # tabel utama + uji berpasangan
streamlit run app_demo.py        # demo hasil

cd ../aplikasi
python3 uji_tampilan.py          # 12 tes tata letak
python3 penyu_live.py            # re-ID realtime dari kamera
python3 penyu_live.py --foto x.jpg   # tanpa kamera
```

Tes dijalankan sebelum mempercayai angka mana pun. Keduanya diverifikasi
dengan mutation testing: tujuh kerusakan disuntikkan sengaja ke `protokol.py`
(kunci sisi dihapus, split dibalik, L2 normalize dilewati, rumus mAP diganti,
dan seterusnya) — ketujuhnya tertangkap.

Bobot MegaDescriptor dibaca dari cache HuggingFace lokal
(`~/.cache/huggingface`), tanpa jaringan.

## Batasan yang harus disebut saat mengutip angka ini

1. **n = 168 terlalu kecil.** Satu prediksi bernilai 0,60 poin; efek harus
   >10 poin baru terdeteksi. Konfirmasi arah datang dari run SeaTurtleIDHeads
   (n = 1.246), yaitu dataset lain.
2. **"Crop kepala" belum benar-benar teruji** — yang diuji center crop 70%
   yang buta. Kepala penyu tidak selalu di tengah frame.
3. **Rank-1 25% berarti 3 dari 4 foto salah di tebakan pertama.** Belum layak
   sebagai penentu identitas otomatis; layak sebagai penyaring kandidat untuk
   diverifikasi manusia.
