# Temuan: MiewID-msv3 mengungguli MegaDescriptor pada re-ID penyu same-side

**Tanggal temuan:** 2026-08-18, 13.50 WITA
**Status:** temuan lintas-repo — **baseline MiewID sudah direproduksi di protokol PCV** (lihat Verifikasi, 14.06 WITA)
**Untuk Notion:** Project → Research for Deep learning → Eksperimen Preprocessing Re-ID Penyu

## Provenance

| Sumber | Identitas |
| --- | --- |
| Repo eksperimen | `PCV` @ `156bd1b9` (branch `main`) |
| Worktree prototipe | `PCV/.worktrees/turtle-reid-prototype` @ `101c7fef` |
| Repo backend | `turtleProject/turtle-identification-be` @ `de7d916` (2026-08-16) |
| Dokumen sumber | `turtle-identification-be/docs/matching-findings.md` (Agu 2026) |
| Commit pemicu | `de7d916` — "Remove MegaDescriptor from the production pipeline" |

Angka PCV di bawah dibaca langsung dari `eksperimen/hasil/*/statistik.json` dan
`eksperimen/hasil/*/rerank_*.json`, bukan diketik ulang dari ingatan.

---

## Ringkasan

1. **MegaDescriptor adalah global ranker yang lemah untuk same-side head photo-ID.**
   Bukan dugaan — dua implementasi independen mengukur angka yang sama.
2. **MiewID-msv3 jauh lebih baik**, dan itu selisih puluhan poin, bukan noise.
3. **Baseline rendah PCV bukan bug.** Checklist "kalau rank-1 di bawah 50 %" di
   `CLAUDE.md` kurang satu item: *modelnya memang salah untuk tugas ini*.
4. **Detektor kepala (YOLO) tetap tidak dibutuhkan untuk Reunion**, dan tidak
   perlu dilatih ulang untuk pack lain — bobot lintas-domain sudah ada. Untuk
   jalur produksi ceritanya lain; lihat [Keputusan](#keputusan--2026-08-18-1353-wita).
5. **Konsekuensi tak menyenangkan:** kalau backbone diganti ke MiewID, eksperimen
   preprocessing di Reunion Green kehilangan hampir seluruh daya ujinya karena
   efek langit-langit.

---

## Temuan 1 — MD lemah, dan angkanya tervalidasi silang

Repo BE mengukur MegaDescriptor-L-384 pada Reunion dengan protokol mereka
sendiri. Hasilnya nyaris identik dengan hasil PCV, padahal implementasi,
kode, dan split-nya berbeda:

| Reunion, same-side | MD — PCV | MD — repo BE | Selisih |
| --- | --- | --- | --- |
| Green (n = 100 query) | 22,0 % | 23,0 % | 1,0 pp |
| Hawksbill (n = 68 query) | 29,4 % | 26,5 % | 2,9 pp |

Jumlah query-nya kebetulan persis sama karena Reunion memang 50 Green + 34
Hawksbill dikali 2 sisi.

**Kenapa ini penting:** selama ini baseline rank-1 25 % di PCV selalu bisa
dicurigai sebagai bug — kebocoran sisi, split terbalik, embedding belum
di-normalize. Sekarang ada implementasi kedua yang tidak tahu apa-apa soal kode
PCV dan sampai di angka yang sama. Implementasi PCV benar. Yang salah adalah
pilihan backbone-nya.

## Temuan 2 — MiewID-msv3 mengubah skalanya

Dari `docs/matching-findings.md`, protokol same-side, `flip=False`, gallery
val split, crop bbox dataset:

| Pack | IDs | n query | MD | **MiewID** | MiewID + XFeat LG CalShortlist |
| --- | --- | --- | --- | --- | --- |
| Reunion Green | 50 | 100 | 23,0 % | **94,0 %** | 94,0 % |
| Reunion Hawksbill | 34 | 68 | 26,5 % | **70,6 %** | 72,1 % |
| Zakynthos | 40 | 80 | 68,8 % | **86,3 %** | 86,3 % |
| Amvrakikos | 50 | 100 | 24,0 % | **56,0 %** | **81,0 %** |

Rata-rata tertimbang MiewID di Reunion (Green + Hawksbill, n = 168):
**≈ 84,5 %**, tanpa matcher lokal apa pun.

Bandingkan dengan pipeline terbaik PCV di dataset yang sama:

| Pipeline PCV (Reunion, n = 168) | rank-1 | rank-5 | mAP |
| --- | --- | --- | --- |
| MD-L-384 raw (baseline) | 25,0 % | 46,4 % | 37,4 |
| MD + rerank geometris XFeat (`murni`, k = 40) | 69,0 % | 76,2 % | 72,7 |

Delta rerank XFeat: **+46,4 pp**, CI95 [38,7 – 54,2], McNemar signifikan.
Itu temuan positif PCV yang paling besar — dan MiewID sendirian, tanpa rerank,
masih di atasnya (≈ 84,5 % vs 69,0 %).

**Baca hati-hati:** split-nya beda. PCV memakai split berbasis tahun (gallery =
tahun pertama, query = tahun kedua), BE memakai `one_per_side` seed=0 (gallery =
val, query = test). Jadi 84,5 % vs 69,0 % **bukan** perbandingan apple-to-apple
dan tidak boleh ditulis sebagai "MiewID mengalahkan pipeline PCV". Yang sah
dinyatakan adalah arah dan besarannya: selisihnya terlalu besar untuk dijelaskan
oleh perbedaan split saja.

## Temuan 3 — kenapa MD gagal di sini

Ringkasan alasan dari dokumen BE, dengan catatan kami:

- MD-L-384 adalah Swin + ArcFace yang dilatih untuk **kemiripan seluruh hewan**,
  dan kuat kalau kemiripan kiri/kanan dieksploitasi lewat **query flip**.
- Protokol paper aslinya ([Adam et al., 2025](https://doi.org/10.1016/j.ecoinf.2025.103158))
  adalah **opposite-side + flip**. Di situ Reunion Green MD ≈ 57 %.
- Protokol kita (dan produksi BE) adalah **same-side, tanpa flip**. Di situ MD
  turun ke 23 %. Selisih 57 % vs 23 % adalah **perbedaan protokol**, bukan bug.
- Same-side head photo-ID pada dasarnya soal **geometri sisik kepala pada satu
  profil**. Descriptor global cosine bukan alat yang tepat; kalau ID benar tidak
  masuk shortlist, matcher lokal tidak bisa menyelamatkan apa pun.
- MiewID-msv3 (EfficientNetV2 440×440, ArcFace atas 64 set re-ID satwa termasuk
  penyu) dilaporkan **+19,2 pp rata-rata top-1** di atas MD-L-384 pada 33 spesies
  yang belum pernah dilihat MD ([Otarashvili et al., 2024](https://arxiv.org/abs/2412.05602)).
  Jadi ini bukan anomali satu eval.

## Temuan 4 — status detektor kepala (YOLO)

Pertanyaan yang memicu penelusuran: kalau model global diganti, apakah YOLO masih
perlu dilatih?

**Bobot yang sudah ada:**

| Bobot | Data latih | Lokasi |
| --- | --- | --- |
| `kepala.pt` (yolo11n, 80 ep, 640 px) | Zakynthos / SeaTurtleID2022 | `eksperimen/yolo_kepala/kepala.pt` |
| `mixed_best.pt` (lintas-domain) | Zakynthos + Amvrakikos (152 train / 28 val) | worktree `yolo_lintas_domain/mixed_best.pt` |

**Detektor domain-tunggal tidak generalisasi** — pola yang sama dengan
MegaDescriptor. Dari `yolo_lintas_domain/cross_domain_detection.json` (n = 180):

| Detektor | recall Zakynthos | recall Amvrakikos | recall overall |
| --- | --- | --- | --- |
| Zakynthos-only (`kepala.pt`) | 93,8 % | **13,0 %** (mean IoU 0,08) | 48,9 % |
| Campuran (`mixed_best.pt`) | 85,0 % | **100 %** | 93,3 % |

Obatnya bukan ganti arsitektur — cukup masukkan domain baru ke data latih.

**Di mana crop kepala benar-benar menolong** (PCV, `hasil/zakynthos_L_squash/`):

| Kondisi Zakynthos (MD-L-384, n = 80) | rank-1 | mAP |
| --- | --- | --- |
| raw baseline | 8,75 % | 21,4 |
| crop YOLO lintas-domain (`kepala_lintas`) | 56,25 % | 66,3 |
| crop kotak anotasi manusia (`kepala_gt`, plafon) | 63,75 % | 74,7 |

Delta `kepala_gt` terhadap raw: **+55,0 pp**, signifikan. Detektor lintas-domain
menangkap ~86 % dari plafon anotasi manusia.

**Di mana crop kepala tidak menolong** — Reunion:

| Kondisi Reunion (MD-L-384, n = 168) | rank-1 |
| --- | --- |
| raw baseline | 25,0 % |
| crop kepala YOLO (`emb_kepala`) | 22,6 % |

Sebabnya kelihatan begitu file-nya dibuka: gambar Reunion **sudah close-up
kepala** (800 × 600, kepala mengisi hampir seluruh frame). Tidak ada latar
berarti untuk dibuang. Repo BE sampai pada kesimpulan yang sama secara
independen: *"Reunion files are already head crops… `kepala.pt` stays off."*
Sama juga dengan SeaTurtleIDHeads (`eksperimen/AUDIT.md:247`).

Kejujuran yang perlu dicatat: crop Reunion itu dibuat pakai `kepala.pt`
(Zakynthos-only), yang di tahap auto-label hanya menemukan kepala di 159 dari 200
foto hijau — 20 % miss. Jadi angka 22,6 % sedikit terkontaminasi detektor buruk.
Arah kesimpulannya tidak berubah, karena plafonnya memang rendah.

**Pipeline detektor Reunion yang setengah jalan:**
`latih_detektor_reunion.py` punya 6 tahap
(`--prepare-annotation → --auto-label → --build → --train → --evaluate → --crop`).
Baru 2 tahap jalan: 336 gambar disiapkan, 159 pseudo-label, review manusia belum
dilakukan. `--train` belum pernah dijalankan. **Rekomendasi: hentikan.**

## Temuan 5 — lubang yang belum tertutup di produksi BE

Semua angka MiewID di dokumen BE memakai `--bbox-crop`, dan itu mengambil
**bbox anotasi dataset**, bukan hasil detektor. Di jalur produksi:

- `app/services/ml/bbox_crop.py` ada.
- `app/services/ml/miewid_embedder.py:63` dan `app/services/ml/xfeat_matcher.py:62`
  sama-sama menerima parameter `bbox`.
- Tapi tidak ada satu pun di `app/api/v1/`, `app/schemas/`, `app/models/`, atau
  `app/services/matching/` yang pernah mengirimkannya.

Artinya foto yang di-upload user masuk ke MiewID **full-frame, tanpa crop**.
Dokumen BE sendiri mencantumkan ini sebagai *Still open #1*.

**Eksperimen yang belum pernah dijalankan siapa pun: MiewID pada foto full-frame
tanpa crop.** Ini yang menentukan apakah detektor masih dibutuhkan sama sekali —
mungkin saja MiewID jauh lebih tahan latar daripada MD. Ujinya murah: jalankan
`eval_miewid_xfeat.py` di Amvrakikos/Zakynthos dengan dan tanpa `--bbox-crop`.

Logika ini sudah tertulis di `eksperimen/protokol.py:154-160` — uji hipotesisnya
lebih dulu, jauh lebih murah daripada membangun detektornya.

---

## Implikasi untuk eksperimen PCV

### Hasil negatif yang lama masih sah, tapi maknanya berubah

"Tidak ada preprocessing yang menaikkan akurasi" tetap benar **untuk
MegaDescriptor-L-384**. Yang sekarang terbuka: apakah nol itu sifat
preprocessing, atau sifat backbone yang lemah? Hasil negatif di backbone yang
kuat adalah klaim yang jauh lebih sulit dibantah.

### Rekomendasi: tambah lengan MiewID, jangan ganti

Hasil MD **jangan dihapus**. Angka MD itu yang membuat angka MiewID punya arti,
dan sekarang ia punya nilai ekstra sebagai validasi silang implementasi PCV.

Secara protokol aman, dan repo kebetulan sudah dirancang untuk ini:

- `protokol.py:531` — pemuatan model terpusat di `muat_model()`; `NAMA_HF` cuma
  dict berkunci env `MODEL`. Menambah MiewID = satu key + satu cabang loader.
- `protokol.py:472` — `dir_hasil()` sudah mempartisi folder per
  `(dataset, model, transform)`. Run MiewID menulis ke `reunion_MIEWID_squash`
  dan tidak bisa mengontaminasi `reunion_L_squash`.

Satu-satunya yang haram: membandingkan MD-raw lawan MiewID-kondisi. Setiap
kondisi wajib punya baseline dari backbone-nya sendiri.

### Peringatan: efek langit-langit membalik prioritas dataset

Di Reunion Green, MiewID 94 % dari 100 query berarti **6 query salah**. Supaya
sebuah kondisi preprocessing terbukti menolong lewat McNemar, ia harus
memperbaiki hampir semua dari 6 itu tanpa merusak satu pun yang sudah benar.
Praktis tidak terdeteksi.

| Pack | MiewID top-1 | query salah | ruang ukur preprocessing |
| --- | --- | --- | --- |
| Amvrakikos | 56 % | ~44 dari 100 | **paling informatif** |
| Reunion Hawksbill | 70,6 % | ~20 dari 68 | cukup |
| Zakynthos | 86 % | ~11 dari 80 | sempit |
| Reunion Green | 94 % | 6 dari 100 | ~nol |

Kebetulan Amvrakikos di PCV memang belum pernah disapu:
`hasil/amvrakikos_L_squash/` hanya berisi `emb_raw`, `emb_kepala_gt`,
`emb_kepala_lintas` — tidak ada `statistik.json`, keenam kondisi belum jalan.

Kalau nanti Green dan Zakynthos tetap dijalankan, "tidak ada delta signifikan"
di sana **wajib dilaporkan sebagai keterbatasan daya uji**, bukan sebagai bukti
preprocessing tidak menolong.

### Yang jangan ikut diadopsi

Jangan tarik XFeat + LighterGlue CalShortlist ke PCV. Biayanya 8–31 detik per
query (bagian *Runtime* di dokumen BE) dan ia menjawab pertanyaan lain —
bagaimana membangun sistem terbaik, bukan apakah preprocessing menolong. Untuk
PCV cukup stage global: MiewID cosine, L2-normalize, nearest neighbor. Perlakuan
persis sama seperti MD sekarang, hanya beda bobot.

### Urutan kerja yang disarankan

1. **Baseline raw MiewID di Reunion dulu** — bukan untuk hasilnya, untuk
   verifikasi. Kalau split-tahun PCV memberi angka jauh di bawah 94 / 71 milik
   BE, itu sendiri temuan (lintas-tahun lebih sulit daripada `one_per_side`) dan
   harus dipahami sebelum lanjut. Ini aturan kerja #2 di `CLAUDE.md`, diterapkan
   ke backbone baru.
2. **Sapu 6 kondisi preprocessing — di Amvrakikos**, bukan Green.
3. Green dan Zakynthos sebagai pelengkap, dilaporkan sebagai *ceiling-limited*.

Biaya: embedding saja, tanpa training, ~336 gambar × 7 kondisi. Tambahan
satu-satunya adalah unduhan bobot MiewID sekali dari HuggingFace — perlu
diperhatikan karena `muat_model()` sekarang sengaja berjalan tanpa jaringan dari
cache HF lokal.

---

## Keputusan — 2026-08-18, 13.53 WITA

**Detektor kepala tetap dibangun.** Alasannya bukan angka eksperimen, melainkan
kondisi pemakaian: di lapangan foto penyu tidak datang dalam keadaan sudah
ter-crop. Semua angka MiewID di dokumen ini memakai bbox anotasi dataset, dan
anotasi itu tidak ada saat seseorang meng-upload foto dari pantai.

Ini menggeser status ablasi "MiewID full-frame vs bbox-crop" dari **go/no-go**
menjadi **pengukuran**: ia tidak lagi menentukan apakah detektor dipakai, tapi
seberapa besar kerugian kalau detektornya meleset — yaitu berapa mahal satu
kesalahan deteksi dibayar dalam poin rank-1. Itu yang menentukan ambang recall
minimum yang harus dipenuhi detektor sebelum layak dipasang.

Konsekuensi yang tetap berlaku dari Temuan 4:

- Reunion tetap tidak butuh detektor (fotonya memang sudah close-up kepala).
- Tidak perlu latih YOLO dari nol — `mixed_best.pt` sudah ada, recall 93,3 %
  overall. Latihan baru hanya kalau foto produksi datang dari domain di luar
  campuran Zakynthos + Amvrakikos.
- Pipeline anotasi Reunion yang setengah jalan tetap dihentikan.

### Hambatan lingkungan yang ditemukan saat cek kelayakan

| Hambatan | Detail |
| --- | --- |
| Data eval BE tidak ada di mesin ini | `eval_miewid_xfeat.py` menunjuk `/Users/abui/@code/sides-matching/data` — tidak ada. Keempat pack ada di `PCV/dataset_penyu/`, jadi eval dijalankan dari sisi PCV, bukan dengan skrip BE apa adanya. |
| `transformers` belum terpasang | `.venv` PCV (Python 3.14.6) belum punya `transformers`; MiewID dimuat lewat `AutoModel.from_pretrained(..., trust_remote_code=True)`. Perlu dipasang sekali. |
| Bobot MiewID belum ter-cache | `muat_model()` sengaja berjalan tanpa jaringan dari cache HF lokal. Unduhan pertama MiewID harus dilakukan eksplisit. |

## Verifikasi — 2026-08-18, 14.06 WITA: MiewID di protokol PCV

Checkpoint 1 selesai. MiewID dijalankan di **protokol PCV sendiri** (split
berbasis tahun, kunci sisi, `TRANSFORM=squash`), bukan protokol BE.

Run: `hasil/reunion_MIEWID_squash/`, `dataset_hash 6a561d9dc6a5791e` — **hash
yang sama persis** dengan run MegaDescriptor `reunion_L_squash`, jadi himpunan
query identik dan uji berpasangan sah.

| Reunion, n = 168 | MD-L-384 | **MiewID-msv3** | Δ |
| --- | --- | --- | --- |
| rank-1 | 25,00 % | **84,52 %** | **+59,52 pp** |
| rank-5 | 46,43 % | **95,24 %** | — |
| mAP | 37,40 | **89,16** | **+51,76** |

- Δ rank-1: CI95 **[51,79 – 67,26]**, signifikan.
- McNemar: n01 = **102** (MD salah → MiewID benar), n10 = **2** (sebaliknya),
  **p = 5,4 × 10⁻²⁸**.
- Δ mAP: CI95 [45,53 – 58,13], signifikan.

n10 = 2 itu yang paling telak: dari 168 query, hanya **dua** yang MD benar tapi
MiewID salah. Ini bukan trade-off, ini dominasi.

**Breakdown wajib:**

| Pecahan | rank-1 | rank-5 | mAP | n |
| --- | --- | --- | --- | --- |
| Green | 95,00 % | 98,00 % | 96,54 | 100 |
| Hawksbill | 69,12 % | 91,18 % | 78,32 | 68 |
| Kiri | 82,14 % | 92,86 % | 86,76 | 84 |
| Kanan | 86,90 % | 97,62 % | 91,57 | 84 |

**Reproduksi angka BE — cocok:**

| Pack | BE (protokol mereka) | PCV (protokol kita) | Selisih |
| --- | --- | --- | --- |
| Reunion Green | 94,0 % | 95,0 % | 1,0 pp |
| Reunion Hawksbill | 70,6 % | 69,1 % | 1,5 pp |

Padahal split-nya beda (tahun vs `one_per_side` seed=0) dan interpolasi
resize-nya beda (`cv2.INTER_AREA` vs `torchvision.Resize` bilinear). Dua
protokol independen sampai ke angka yang sama — kesimpulan Temuan 2 berdiri.

Satu hal yang membuat angka kita **lebih kuat**, bukan lebih lemah: gallery PCV
per sisi berisi 84 foto (50 Green + 34 Hawksbill), sedangkan gallery Green di BE
hanya 50 foto Green. Query Green kita bersaing melawan distraktor lintas
spesies dan tetap mendapat 95,0 %.

### Catatan implementasi

- `AutoModel.from_pretrained(..., trust_remote_code=True)` — jalur resmi BE —
  **gagal di transformers 5.15**: remote code MiewID ditulis untuk 4.45
  (`AttributeError: 'MiewIdNet' object has no attribute 'all_tied_weights_keys'`).
  Solusi di `protokol.py:_muat_miewid()`: kelas hulu diimpor langsung dari
  snapshot, bobot dimuat manual dengan `strict=True` — pola yang sama dengan
  cabang MegaDescriptor. Bukan tulisan ulang arsitektur.
- MegaDescriptor tidak tersentuh. `MODEL=L` tetap 384 / 1536 / crop_pct 0,9,
  dan `dir_hasil()` menulis MiewID ke folder terpisah.
- Waktu: 336 gambar, CPU, ~3,5 menit.

**Konsekuensi:** efek langit-langit yang diperingatkan di atas **terkonfirmasi
di Green** — 95 % dari 100 query berarti hanya 5 query salah. Sapuan
preprocessing di Green praktis tidak punya daya uji. Hawksbill (69,1 %, 21
query salah) masih punya ruang, dan Amvrakikos tetap yang paling informatif.

## Hasil Amvrakikos — 2026-08-18, 14.30 WITA

Katalog Amvrakikos di-port dari worktree ke `protokol.py` utama. Split sehat:
100 gallery / 100 query, 50 identitas, kiri 50 / kanan 50, satu orientasi
`top` dibuang. Regresi `reunion` (168/168) dan `zakynthos` (80/80) aman.

### Sapuan preprocessing — hasil negatif bertahan di backbone kuat

| Kondisi (MiewID, n = 100) | rank-1 | Δ rank-1 [CI95] | McNemar p | Δ mAP |
| --- | --- | --- | --- | --- |
| **raw (baseline)** | **57,00** | — | — | — |
| Crop tengah 70 % | 55,00 | −2 [−9, +5] | 0,791 | −2,77 ns |
| White balance | 56,00 | −1 | — | −1,68 ns |
| CLAHE | 55,00 | −2 [−9, +5] | 0,791 | −1,07 ns |
| Grayscale | 54,00 | −3 [−10, +4] | 0,549 | −0,65 ns |
| Crop + WB | 53,00 | −4 [−11, +3] | 0,424 | −4,11 ns |
| Resize 368 | 53,00 | −4 [−9, 0] | 0,219 | −2,01 ns |

Semua delta negatif, tidak satu pun signifikan — di pack dengan ruang ukur
terlebar (43 query salah) dan dengan backbone yang kuat. **Hasil negatif
eksperimen ini bukan artefak MegaDescriptor.** Itu klaim yang jauh lebih sulit
dibantah daripada yang kita punya sebelumnya.

### Ablasi crop — di kedua backbone, tidak signifikan

| Amvrakikos, n = 100 | MD-L-384 | MiewID | Δ backbone [CI95] | p |
| --- | --- | --- | --- | --- |
| Tanpa crop (full-frame) | 16,0 % | **57,0 %** | +41 [+30, +52] | 2,5 × 10⁻¹⁰ |
| Dengan crop (kotak manusia) | 23,0 % | **61,0 %** | +38 [+27, +49] | 5,1 × 10⁻⁹ |

| Efek crop di dalam satu backbone | raw → crop | Δ [CI95] | p |
| --- | --- | --- | --- |
| MegaDescriptor | 16,0 → 23,0 | +7 [−1, +16] | 0,167 ns |
| MiewID | 57,0 → 61,0 | +4 [−4, +12] | 0,481 ns |

Tiga bacaan:

1. **Crop tidak signifikan di kedua backbone** pada pack ini. Yang dulu membuat
   crop tampak sakti adalah Zakynthos (+55 pp) — kasus ekstrem, penyu kecil di
   frame besar. Amvrakikos tidak begitu.
2. **MiewID tanpa crop (57 %) mengalahkan MegaDescriptor dengan crop sempurna
   (23 %)** — selisih 34 poin. Ganti backbone jauh mengungguli tambah detektor.
3. Keuntungan crop menyusut saat backbone menguat (+7 → +4), konsisten dengan
   "MiewID lebih tahan latar". Dua-duanya ns, jadi selisih antar-selisih itu
   sendiri tidak boleh diklaim.

**Peringatan daya uji:** n = 100 dan CI crop MiewID [−4, +12] itu lebar. Data ini
**gagal membuktikan crop menolong**; ia tidak membuktikan crop tidak berguna.

### Preprocessing di atas crop — celah terakhir ditutup (15.07 WITA)

Bantahan yang masih mungkin diajukan terhadap hasil negatif kita:

> Wajar nol. Kamu mengukur white balance dan CLAHE di foto penuh, dan yang
> mendominasi foto itu **latar**. Statistik gray-world dan histogram CLAHE
> dihitung dari seluruh piksel, jadi yang "diperbaiki" justru latarnya. Buang
> dulu latarnya, baru ukur.

Argumen itu masuk akal, dan sampai sekarang tidak terjawab. Untuk mengujinya,
`embed()` diberi kondisi **komposit** `berkas+array` (mis. `kepala_gt+clahe`):
potong kepala dulu, baru terapkan transform di atas potongannya — jadi
statistik preprocessing dihitung dari penyunya, bukan dari pasir dan air.
Urutan crop-dulu itu sengaja, dan justru versi yang paling mungkin berhasil.

Baseline di sini **`kepala_gt`**, bukan `raw` — pertanyaannya "apakah
preprocessing menolong setelah latar dibuang".

| Amvrakikos, MiewID, n = 100 | rank-1 | rank-5 | mAP | Δ vs `kepala_gt` [CI95] | p |
| --- | --- | --- | --- | --- | --- |
| raw (full-frame) | 57,00 | 81,00 | 67,33 | — | — |
| **`kepala_gt`** (baseline crop) | **61,00** | 80,00 | 71,01 | +4,00 vs raw | 0,481 |
| `kepala_gt+wb` | 62,00 | 81,00 | 70,78 | **+1,00** [−2, +4] | 1,000 |
| `kepala_gt+clahe` | 56,00 | 83,00 | 68,44 | **−5,00** [−11, +1] | 0,180 |
| `kepala_gt+gray` | 57,00 | 84,00 | 69,22 | **−4,00** [−11, +3] | 0,388 |
| `kepala_gt+resize368` | 60,00 | 82,00 | 70,97 | **−1,00** [−5, +2] | 1,000 |

**Nol lagi, dan celahnya tertutup.** Tiga dari empat justru negatif; ΔmAP
keempatnya negatif atau nol. Tidak satu pun signifikan.

Jadi hasil negatif eksperimen ini sekarang bertahan terhadap tiga bantahan
sekaligus: bukan karena backbone lemah (diuji dengan MiewID), bukan karena
dataset tanpa ruang ukur (diuji di pack dengan 43 query salah), dan bukan
karena latar mengganggu pengukuran (diuji di atas crop kepala).

Catatan yang **tidak** boleh diklaim: rank-5 naik di `clahe` (83) dan `gray`
(84) sementara rank-1 turun. Itu 3–4 query pada n = 100. Bukan temuan.

Pembanding lama, crop tengah 70 % + white balance: 55,0 → 53,0,
Δ −2 [−7, +3], p = 0,727. Sama saja.

## Keputusan 2 — 2026-08-18, 14.47 WITA: crop + 440×440 ditetapkan

Pipeline standar aplikasi ditetapkan: **crop kepala → 512×512 → MiewID 440×440**.

Dasarnya kondisi pemakaian, bukan signifikansi statistik — angka di atas
menunjukkan crop hanya +4 pp dan tidak signifikan. Itu sudah disampaikan dan
keputusannya tetap: di lapangan foto tidak datang ter-crop, dan pipeline harus
punya satu jalur yang pasti.

### Yang dikerjakan

- `mixed_best.pt` disalin dari worktree ke `eksperimen/yolo_kepala/kepala_lintas.pt`
  supaya repo utama tidak bergantung pada worktree yang di-ignore git.
  Dipakai **bukan** `kepala.pt` — recall lintas-domain 93,3 % vs 48,9 %.
- `aplikasi/penyu_live.py`: tahap DETECT sekarang YOLO kepala; kontur terbesar
  turun jadi cadangan. Ditambah `deteksi_kepala()` dan `potong_kepala()`.
- Kondisi crop kini bisa dipakai **kamera langsung**. Sebelumnya kondisi
  berbasis-berkas selalu menolak frame kamera karena tidak ada anotasi.
- `potong_kepala()` menyalin geometri `P.kepala_gt()` persis — margin 18 %,
  resize `INTER_AREA` ke 512 — supaya potongan kamera dan potongan galeri
  hidup di distribusi yang sama.
- Kalau kepala tidak terdeteksi, jalurnya **melempar galat, bukan jatuh ke
  frame penuh**. Query tanpa crop dibandingkan dengan galeri ter-crop
  menghasilkan cosine ~0,05 tanpa satu pun pesan error — bug itu pernah
  terjadi di repo ini.

### Verifikasi

Foto Amvrakikos `01_2017_left_DSC03532.JPG` (5152 × 3864):

| | x | y | w | h |
| --- | --- | --- | --- | --- |
| Kotak YOLO | 1280 | 1190 | 2464 | 1724 |
| Kotak anotasi manusia | 1283 | 1221 | 2475 | 1677 |

Selisih ~30 px pada gambar 5152 px. Potongan keluar `512 × 512 × 3`, dan
setelah transform kanonik menjadi `(3, 440, 440)` — ukuran input MiewID
terkonfirmasi.

Catatan: 440 × 440 tidak perlu "ditetapkan" secara terpisah — ia sudah otomatis
mengikuti `MODEL=MIEWID` lewat cabang `UKURAN` di `protokol.py`, dan `TRANSFORM`
tetap `squash` (tanpa center-crop, `CROP_PCT = 1.0`).

## Sapuan Reunion dengan MiewID — 2026-08-18, 15.22 WITA

Baseline `raw` di sini 84,52 % (bukan 25,00 % milik MegaDescriptor), jadi
variansinya jauh lebih kecil dan efek yang dulu tenggelam jadi terlihat.

| Kondisi (MiewID, n = 168) | R-1 | R-5 | mAP | Δ R-1 [CI95] | p | Δ mAP |
| --- | --- | --- | --- | --- | --- | --- |
| **Raw (baseline)** | **84,52** | 95,24 | 89,16 | — | — | — |
| CLAHE | 85,71 | 94,05 | 89,27 | +1,19 [−1,8, +4,2] | 0,688 | +0,10 ns |
| Resize 368 | 84,52 | 95,83 | 89,24 | 0,00 [−1,8, +1,8] | 1,000 | +0,08 ns |
| Grayscale | 83,33 | 94,64 | 88,15 | −1,19 [−4,2, +1,8] | 0,688 | −1,01 ns |
| White balance | 81,55 | 92,86 | 87,02 | −2,98 [−7,1, +0,6] | 0,227 | −2,14 ns |
| Crop tengah 70 % | 79,76 | 90,48 | 84,56 | −4,76 [−9,5, 0,0] | 0,077 | **−4,60 SIG** |
| **Crop + WB** | **73,21** | 87,50 | 79,04 | **−11,31 [−17,3, −5,4]** | **0,00031** | **−10,12 SIG** |

**Ini temuan positif pertama dengan arah negatif.** `crop_wb` merusak 11 poin
dengan p = 0,00031 — bukan noise, dan lolos uji dengan margin lebar. `crop`
sendiri merusak mAP secara signifikan.

Di MegaDescriptor semua kondisi hanyut di noise pada baseline 25 %; tidak ada
yang signifikan ke arah mana pun. Dengan backbone kuat, kerusakannya terukur.
Jadi kesimpulan naik kelas: **preprocessing di Reunion bukan sekadar tidak
berguna — sebagiannya merusak, dan itu bisa dibuktikan.**

Breakdown rank-1:

| Kondisi | kiri | kanan | Green | Hawksbill |
| --- | --- | --- | --- | --- |
| Raw | 82,14 | 86,90 | 95,00 | 69,12 |
| CLAHE | 84,52 | 86,90 | 94,00 | 73,53 |
| Crop + WB | 65,48 | 80,95 | 79,00 | 64,71 |

Efek langit-langit yang dikhawatirkan ternyata hanya menutup ruang untuk
**kenaikan**, bukan untuk **penurunan** — Green 95,00 → 79,00 terlihat jelas.

### Mekanisme kerusakan (penalaran, bukan pengukuran)

Pemeriksaan visual satu query yang berubah dari benar ke salah
(`Green/Dils` kiri; 23 dari 168 query berubah begitu):

1. **`crop` membuang sisik tepi.** Foto Reunion sudah close-up kepala, jadi
   memotong tengah 70 % memotong ubun-ubun dan mendorong mata ke pinggir.
   Yang dibuang bukan latar, melainkan penanda identitas.
2. **`wb` menghancurkan warna.** Gray-world menganggap rata-rata frame
   seharusnya abu-abu netral; foto bawah laut didominasi biru-hijau, jadi
   koreksinya mendorong jauh ke merah.

Galeri diperlakukan sama, jadi ini bukan ketidakcocokan distribusi. Hipotesis
mekanismenya: **besarnya koreksi berbeda tiap foto**, mengikuti kedalaman,
kejernihan air, dan seberapa besar penyu mengisi frame — jadi transform yang
niatnya menyeragamkan justru menyuntikkan variasi yang mengikuti kondisi
pemotretan. Crop memperparah karena rata-rata dihitung dari wilayah lebih
kecil yang lebih didominasi penyu. Itu konsisten dengan `crop_wb` (−11,31)
jauh lebih rusak daripada `crop` (−4,76) atau `wb` (−2,98) sendiri.

Belum diuji. Ujinya: bandingkan sebaran faktor koreksi gray-world antar foto
di `raw` vs `crop`.

---

## Sapuan Zakynthos dengan MiewID — 2026-08-18, 15.40 WITA

| Kondisi (MiewID, n = 80) | R-1 | mAP | Δ R-1 [CI95] | p |
| --- | --- | --- | --- | --- |
| Raw (baseline) | 35,00 | 46,71 | — | — |
| Resize 368 | 28,75 | 42,68 | −6,25 [−12,5, −1,3] | 0,063 (ΔmAP **−3,79 SIG**) |
| CLAHE | 32,50 | 45,18 | −2,50 | 0,727 ns |
| Grayscale | 36,25 | 46,21 | +1,25 | 1,000 ns |
| White balance | 37,50 | 46,47 | +2,50 | ns |
| **Crop tengah 70 %** | **55,00** | 63,73 | **+20,00 [+11,3, +30,0]** | **0,000145** |
| **Crop + WB** | **55,00** | 63,16 | **+20,00** | **0,000145** |
| **`kepala_gt`** | **88,75** | 91,44 | **+53,75 [+42,5, +65,0]** | **2,6 × 10⁻¹²** |

**Koreksi terhadap catatan sebelumnya:** klaim "crop buta selalu merugikan"
SALAH. Di Zakynthos crop tengah 70 % naik 20 poin dan signifikan — kondisi
preprocessing positif-signifikan pertama di seluruh eksperimen ini.

Dugaan bahwa backbone kuat akan menghapus keuntungan crop juga **salah**:
+55,00 (MD) vs +53,75 (MiewID), praktis tidak berubah.

### Fungsi crop: memulihkan resolusi, bukan membuang latar

Model menerima input tetap 440 × 440. Kalau seluruh frame diperas ke sana,
kepala ikut mengecil sesuai porsinya. Diukur dari kotak anotasi:

| Pack | Kepala % luas frame | Ukuran kepala setelah frame diperas ke 440 | Δ crop kepala |
| --- | --- | --- | --- |
| Zakynthos | **2,1 %** | **~64 × 64 px** | **+53,75 SIG** |
| Amvrakikos | 35,1 % | ~261 × 261 px | +4,00 ns |
| Reunion | ~seluruh frame (visual) | ~440 × 440 px | −2,4 (merusak) |

Urutannya monoton. Di Zakynthos model hanya melihat kepala 64 × 64 piksel —
pola sisik hancur sebelum sempat dilihat. Crop mengembalikannya ke 440 penuh.
Crop buta menangkap ~⅓ dari keuntungan itu (luas frame jadi separuh, kepala
~91 px) asal penyunya dekat tengah. `resize368` merusak karena mengecilkan
kepala yang sudah cuma 64 px.

### Dua jenis preprocessing, dan hanya satu yang berarti

| Jenis | Kondisi | Hasil |
| --- | --- | --- |
| **Warna / normalisasi** | white balance, CLAHE, grayscale | **Nol di mana-mana** — 2 backbone, 3 dataset, di frame penuh maupun di atas crop kepala. Tidak satu pun signifikan positif. |
| **Framing / resolusi** | crop kepala | Ditentukan sepenuhnya oleh seberapa kecil kepala di frame. +54 / +4 / −2. |

Jawaban eksperimen ini bukan "preprocessing tidak menolong", melainkan:
**yang menolong hanya memberi model piksel kepala yang cukup. Sisanya nol.**

## Re-ranking XFeat di atas MiewID — 2026-08-18, 15.53 WITA

Menguji apakah re-ranking lokal mentah merusak ranking global yang sudah
bagus. Kode yang dipakai **sama persis** dengan yang dulu memberi +46,4 pp di
atas MegaDescriptor — matcher sama, k sama, skor inlier RANSAC sama. Satu-satunya
yang berubah: mutu stage-1.

Reunion, `xfeat`, k = 40, n = 168 (6.720 pasangan, 7,8 ms/pasangan):

| | R-1 | R-5 | mAP | Δ R-1 [CI95] | p |
| --- | --- | --- | --- | --- | --- |
| stage-1 (MiewID) | **84,52** | 95,24 | 89,16 | — | — |
| `murni` (rerank XFeat) | **50,00** | 62,50 | 56,94 | **−34,52 [−42,9, −26,2]** | 2,4 × 10⁻¹³ |
| `rrf` (fusi peringkat) | 60,71 | 77,98 | 69,09 | **−23,81 [−31,6, −16,1]** | 1,0 × 10⁻⁸ |

Per spesies: Green 95,00 → 53,00, Hawksbill 69,12 → 45,59.

**Mode kegagalan BE tereproduksi di protokol kita.** Mereka mencatat MiewID
Green 94 → 44 dengan MNN mentah; kita dapat 95,00 → 53,00. Implementasi
berbeda, kesimpulan sama.

Dan `rrf` — fusi peringkat naif — **tetap kalah 23,8 poin**. Jadi tidak cukup
sekadar "difusikan"; yang dibutuhkan fusi **skor terkalibrasi** (isotonic +
PCHIP atas cosine dan atas skor lokal), seperti CalShortlist di repo BE.

**Pelajarannya:** nilai re-ranking lokal adalah fungsi dari seberapa buruk
model global. Di MegaDescriptor (25,0 %) ia +46,4 pp. Di MiewID (84,5 %) ia
−34,5 pp. Matcher yang sama.

## Fusi terkalibrasi (CalShortlist) — 2026-08-18, 16.05 WITA

Dibangun di `eksperimen/calshortlist.py`. Alih-alih **mengganti** urutan
stage-1, kedua skor dipetakan ke probabilitas lebih dulu lewat isotonic
regression, baru dirata-ratakan: `f = 0.5 (p_global + p_lokal)`. Kalibrator
dipasang **leave-one-query-out** supaya query yang dinilai tidak pernah ikut
melatih kalibratornya sendiri.

Memakai skor XFeat yang **sudah ada** dari run rerank — tidak ada matching
yang dihitung ulang.

Reunion, `xfeat`, k = 40, n = 168:

| | R-1 | R-5 | mAP | Δ R-1 vs stage-1 [CI95] | p |
| --- | --- | --- | --- | --- | --- |
| stage-1 (MiewID) | **84,52** | 95,24 | 89,16 | — | — |
| `lokal` (rerank mentah) | 50,00 | 62,50 | 56,75 | −34,52 [−42,9, −26,2] | 2,4 × 10⁻¹³ |
| **`cal` (fusi terkalibrasi)** | **81,55** | 94,64 | 86,98 | **−2,98 [−7,1, +1,2]** | **0,267 ns** |

**Kalibrasi menyelamatkan kerusakan, tapi tidak menambah apa pun.** Dari
−34,52 (signifikan) jadi −2,98 (tidak signifikan). Di Reunion, MiewID sendirian
tetap yang terbaik.

Ini persis pola yang dilaporkan repo BE: CalShortlist **tidak** meregresi
Green/Zakynthos tapi juga tidak menaikkannya; keuntungannya hanya muncul di
Amvrakikos (+25 pp), satu-satunya pack di mana stage-1 masih lemah.

Belum diuji di sini: CalShortlist pada Amvrakikos (butuh run rerank XFeat
untuk pack itu lebih dulu), dan matcher LighterGlue sebagai ganti XFeat+RANSAC.

## Pipeline akhir dan dua uji yang menutupnya — 2026-08-18, sore

### Ambang detektor: 0,25 terlalu ketat

Zakynthos, 15 dari 160 foto (9 %) tidak menghasilkan deteksi di ambang 0,25;
Amvrakikos nol. Dari 15 itu, 12 terdeteksi di ambang 0,02 dan 9 berkotak bagus
(IoU 0,69–0,90) — detektornya melihat kepalanya, skornya saja di bawah ambang.
Kualitas kotak sendiri sudah setara manusia (IoU median 0,932, rasio luas
median 1,00×), jadi ini **bukan** soal detektor kurang dilatih.

| Zakynthos, n = 80 | R-1 | mAP |
| --- | --- | --- |
| raw | 35,00 | 46,47 |
| potongan lama (`kepala_lintas`) | 76,25 | 80,89 |
| pipeline baru @ 0,25 | **76,25** | 80,92 |
| pipeline baru @ 0,05 | **82,50** | 85,80 |
| kotak anotasi manusia | 88,75 | 91,44 |

- pipeline baru @0,25 vs potongan lama: Δ 0,00 [0,00, 0,00] — **pipeline tervalidasi**
- 0,25 → 0,05: **+6,25** [+1,25, +12,50], McNemar **p = 0,0625** (di ambang)
- 0,05 → anotasi: +6,25 [−1,25, +13,75], p = 0,18 — **tidak lagi signifikan**

`KONF_KEPALA` di `aplikasi/penyu_live.py` diubah 0,25 → 0,05.

Temuan sampingan: potongan lama diam-diam memakai **frame penuh** sebagai
fallback untuk 15 foto gagal (sudah diverifikasi identik dengan squash 512).
Jadi angka `kepala_lintas` selama ini adalah "detektor **plus** fallback".

### Uji 1 — crop kepala di Reunion: netral, bukan merusak

| Reunion, n = 168 | R-1 | R-5 | mAP |
| --- | --- | --- | --- |
| raw | 84,52 | 95,24 | 89,16 |
| crop kepala YOLO @0,05 | 85,71 | 94,64 | 89,76 |

Δ **+1,19** [−2,38, +4,76], p = 0,754; ΔmAP +0,59 ns. (55 dari 336 foto gagal
deteksi dan jatuh ke frame penuh — tidak berbahaya di sini, karena frame penuh
Reunion memang sudah berupa kepala.)

Yang dulu merusak Reunion adalah **crop tengah 70 % buta** (−4,76), bukan crop
ke kotak kepala. Dua hal berbeda. **Konsekuensi: pipeline tidak butuh
percabangan untuk crop — selalu crop kalau deteksi berhasil.**

### Uji 2 — gerbang margin: berguna untuk biaya, bukan akurasi

Gerbang: margin stage-1 (cosine #1 − #2) ≥ ambang → pakai stage-1; di bawahnya
→ jalankan matcher lokal + fusi terkalibrasi. Ambang dipilih
**leave-one-query-out**; angka oracle dicetak terpisah dan ditandai.

| | stage-1 | lokal terkalibrasi | **LOO gerbang** | oracle |
| --- | --- | --- | --- | --- |
| Reunion (n=168) | **84,52** | 81,55 | 82,74 | 85,12 |
| Amvrakikos (n=100) | 57,00 | **90,00** | **90,00** | 90,00 |

Di Reunion gerbang mengurangi kerusakan (−2,98 → −1,79) tapi tidak
membalikkannya, dan **oracle-nya cuma 85,12 % — 0,6 poin di atas stage-1, satu
query.** Artinya tidak ada ambang yang bisa membuat pencocokan lokal menang di
Reunion. Di Amvrakikos gerbang selalu menyala, dan itu benar.

Nilainya ada di **biaya**: 91 % query Reunion melewati matcher yang harganya
~6 detik, dan akurasinya justru sedikit lebih baik daripada memaksa lokal ke
semua query (82,74 vs 81,55).

### Pipeline yang dihasilkan

Tidak ada percabangan berdasarkan nama dataset — foto lapangan tidak berlabel.

1. **Deteksi kepala** (ambang 0,05). Berhasil → crop 18 % margin, 512×512.
   Gagal → **tolak**, jangan pakai frame penuh.
2. **Embedding MiewID 440×440**, cosine ke galeri, terkunci sisi.
3. **Gerbang margin**: margin besar → selesai (< 1 detik). Margin kecil →
   LighterGlue top-k + fusi terkalibrasi (~6 detik).
4. **Kalau ada dua sisi**: aturan `fallback` — rank-1 sepakat pakai itu, kalau
   tidak pakai yang margin-nya lebih besar (+4,76 pp; +7,35 di Hawksbill).

---

## Kesimpulan

1. **MegaDescriptor adalah backbone yang salah untuk tugas ini.** MiewID
   menaikkan rank-1 Reunion 25,00 → 84,52 (p = 5,4 × 10⁻²⁸), Amvrakikos
   16,00 → 57,00 (p = 2,5 × 10⁻¹⁰), Zakynthos 8,75 → 35,00 raw dan
   63,75 → 88,75 dengan crop kepala. Tervalidasi silang dengan implementasi
   independen di repo BE.

2. **Preprocessing warna/normalisasi tidak berguna, di mana pun.** White
   balance, CLAHE, grayscale: nol di tiga dataset, dua backbone, di frame penuh
   maupun di atas crop kepala. Tidak satu pun signifikan positif. Ini hasil
   negatif yang sekarang sangat kokoh.

3. **Yang menolong hanya resolusi kepala.** Keuntungan crop ditentukan
   sepenuhnya oleh seberapa kecil kepala di frame: Zakynthos (2,1 % frame,
   ~64 px) **+53,75 SIG**; Amvrakikos (35,1 %, ~261 px) +4,00 ns; Reunion
   (hampir seluruh frame) −2,4 merusak. Bukan "membuang latar" — memulihkan
   piksel.

4. **Sebagian preprocessing merusak, dan itu baru terlihat setelah backbone
   diperbaiki.** Reunion `crop_wb` −11,31 (p = 0,00031); Zakynthos `resize368`
   ΔmAP −3,79 signifikan. Di MegaDescriptor semuanya tenggelam di noise.

5. **Nilai re-ranking lokal adalah fungsi dari seberapa buruk model global.**
   Kode yang sama: +46,4 pp di atas MegaDescriptor, −34,5 pp di atas MiewID.
   Fusi terkalibrasi menetralkan kerusakan (−2,98 ns) tapi tidak menambah nilai
   di pack yang stage-1-nya sudah kuat.

6. **Detektor kepala tetap dibangun**, dan sekarang ada dasar datanya, bukan
   hanya alasan lapangan: di framing tipe Zakynthos — yang paling mirip foto
   lapangan sungguhan — crop bernilai ~54 poin. Sudah terpasang di aplikasi dan
   terverifikasi end-to-end.

### Yang TIDAK boleh disimpulkan dari sini

- Semua angka bersifat **closed-set**: jawaban benar selalu ada di galeri.
  Tidak ada ambang penolakan, tidak ada pengukuran false-accept untuk penyu
  yang belum pernah terdaftar. Untuk pemakaian lapangan, ini lubang terbesar.
- Semua dataset adalah dataset riset Yunani dan Reunion. Belum ada satu pun
  foto lapangan Indonesia yang diuji.
- Uji end-to-end aplikasi baru 12 query.

## Yang belum diukur (jangan diklaim sebelum dijalankan)

- MiewID pada foto full-frame tanpa crop bbox.
- Apakah keenam kondisi preprocessing tetap nol di bawah MiewID.
- Dual L+R pair matching di Amvrakikos / Zakynthos — dokumen BE hanya menguji di
  Reunion dan memperingatkan agar +13 pp Hawksbill tidak diasumsikan menular.

## Referensi

- Adam, L., Papafitsoros, K., Jean, C., Rees, A. F., & Čermák, V. (2025).
  Exploiting facial side similarities to improve AI-driven sea turtle
  photo-identification systems. *Ecological Informatics*, 89, 103158.
  <https://doi.org/10.1016/j.ecoinf.2025.103158>
- Otarashvili, L., et al. (2024). MiewID — multispecies wildlife re-identification.
  <https://arxiv.org/abs/2412.05602>
- Čermák, V., Picek, L., Adam, L., & Papafitsoros, K. (2023). WildlifeDatasets.
  <https://arxiv.org/abs/2311.09118>
- Cermak, V., et al. (2024). WildFusion: calibrated similarity fusion.
  <https://arxiv.org/abs/2408.12934>
- Yesharim, Y., et al. (2026). Near-perfect photo-ID of the Hula painted frog
  with zero-shot deep local-feature matching. *Ecological Informatics*, 98, 103942.
  <https://doi.org/10.1016/j.ecoinf.2026.103942>
- `turtle-identification-be/docs/matching-findings.md`
