# PCV — Eksperimen Preprocessing Re-ID Penyu

## Konteks

Ini **eksperimen, bukan produk**. Satu pertanyaan yang dijawab:
preprocessing mana yang benar-benar menaikkan akurasi re-identifikasi penyu,
dan seberapa besar?

Spesifikasi lengkap ada di `docs/eksperimen-preprocessing-reid-penyu.md`.
Baca file itu sebelum menyentuh kode. Kalau ada konflik antara file ini dan
spesifikasi itu, spesifikasi yang menang.

- Dataset: ReunionTurtles (84 individu, 4 foto/individu, ~336 gambar)
- Model: MegaDescriptor, **frozen**, tanpa fine-tuning
- Identifikasi: embedding + nearest neighbor, bukan training

## Yang TIDAK dikerjakan di repo ini

- Membangun aplikasi atau API produksi
- Fine-tuning / training model
- Optimasi kecepatan inferensi
- Handling input kamera real-time

Kecuali: satu halaman Streamlit tipis sebagai alat demo. Itu bukan produk.

## Protokol yang dikunci — jangan diubah tanpa izin

Semua kondisi preprocessing wajib memakai setup identik. Kalau satu saja
berubah, perbandingan batal.

- Gallery = semua foto tahun pertama. Query = semua foto tahun kedua.
- Split deterministik berbasis tahun. **Tidak ada random split, tidak ada seed.**
- Matching: cosine similarity atas embedding yang sudah di-L2 normalize.
- Query sisi kiri hanya dicari di gallery kiri. Kanan hanya ke kanan.
  **Tidak boleh dicocokkan silang.**
- Bobot dan versi MegaDescriptor sama persis di semua run.

Catat versi model, versi library, dan hash dataset di header tiap notebook/run.

## Aturan kerja

1. Kerjakan berurutan. **Berhenti dan lapor di tiap checkpoint** sebelum lanjut.
2. Jangan sentuh preprocessing sebelum baseline raw jalan DAN metrik manual
   sudah cocok dengan `wildlife-tools`.
3. Satu kondisi preprocessing per satu. Jangan gabung.
4. Jangan pernah melaporkan dua angka telanjang untuk dibandingkan dengan mata.
   Selalu delta terhadap baseline + confidence interval.

## Aturan statistik — ini yang paling sering dilanggar

Jumlah query hanya ~84–168. **Satu prediksi berubah ≈ 1 poin persen.**

> Selisih 3–4 poin antar kondisi preprocessing adalah **noise, bukan temuan.**

Karena himpunan query identik di semua kondisi, gunakan paired comparison:

- McNemar's test untuk rank-1 (data berpasangan biner)
- Bootstrap confidence interval untuk selisih mAP

Jangan menulis "kondisi X lebih baik" tanpa uji statistik yang mendukungnya.

## Kalau baseline rank-1 di bawah 50%

Jangan salahkan preprocessing. Curigai dulu, berurutan:

1. Kebocoran sisi (kiri dicocokkan ke kanan)
2. Split gallery/query terbalik
3. Embedding belum di-normalize
4. Ukuran input atau normalisasi tidak sesuai ekspektasi model
5. Ada individu di query yang tidak punya pasangan di gallery

## Metrik yang dilaporkan

Rank-1, Rank-5, mAP. Rank-5 tetap dilaporkan karena di praktik konservasi
manusia melakukan verifikasi akhir.

Wajib ada breakdown per **spesies** (hijau vs sisik) dan per **sisi**
(kiri vs kanan). Temuan menarik biasanya muncul di sini, bukan di angka gabungan.

## Dokumentasi

Setelah tiap checkpoint, tulis progres dan temuan ke Notion:
**Project → Research for Deep learning → Eksperimen Preprocessing Re-ID Penyu**

Halaman itu sudah punya kerangka toggle. Isi toggle yang sesuai, jangan bikin
struktur baru.

## Gaya komunikasi

- Bahasa Indonesia.
- Kalau angka mencurigakan, bilang mencurigakan. Jangan dipoles.
- Hasil negatif (preprocessing menurunkan akurasi) adalah temuan yang valid
  dan harus dilaporkan apa adanya, bukan disembunyikan atau di-tuning sampai
  kelihatan bagus.
