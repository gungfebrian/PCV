# PENJELASAN FUNGSI titik() - MENGURUTKAN 4 POJOK KARTU

## Visualisasi Koordinat Kartu

```
       Y
       ↑
       |
   (0,0) ──────────────→ X
       |
```

## Contoh 4 Pojok Kartu yang Miring

```
   Pojok A (50, 30)      Pojok B (350, 40)
         ● ─────────────────●
         │                   │
         │                   │
         │     KARTU         │
         │                   │
         │                   │
         ●                   ●
   Pojok C (60, 400)     Pojok D (340, 420)
```

## Data Pojok (Tidak Terurut)

```python
pts = [
    [50, 30],    # Pojok A
    [350, 40],   # Pojok B  
    [60, 400],   # Pojok C
    [340, 420]   # Pojok D
]
```

## STEP 1: Hitung x + y (Sum)

```python
s = pts.sum(axis = 1)
# s = [80, 390, 460, 760]
#     └A─┘ └B─┘ └C─┘ └D─┘
```

**Kenapa?**
- x + y kecil → dekat ke (0,0) → **Kiri Atas**
- x + y besar → jauh dari (0,0) → **Kanan Bawah**

```
   A (50+30=80) ← TERKECIL → Kiri Atas ✅
   B (350+40=390)
   C (60+400=460)
   D (340+420=760) ← TERBESAR → Kanan Bawah ✅
```

## STEP 2: Hitung x - y (Diff)

```python
diff = np.diff(pts, axis = 1)
# diff = [20, 310, -340, -80]
#        └A─┘ └B─┘ └C──┘ └D─┘
```

**Kenapa?**
- x - y menunjukkan "keseimbangan" x vs y
- Pojok yang tersisa perlu dibedakan: **Kanan Atas** vs **Kiri Bawah**

```
   A: 50-30=20   (positif kecil)
   B: 350-40=310 (positif BESAR - lebih ke kanan)
   C: 60-400=-340 (negatif BESAR - lebih ke bawah)
   D: 340-420=-80 (negatif kecil)
```

## Hasil Akhir

```python
rect[0] = pts[np.argmin(s)]    # x+y terkecil → A (50, 30) = Kiri Atas ✅
rect[2] = pts[np.argmax(s)]    # x+y terbesar → D (340, 420) = Kanan Bawah ✅
rect[1] = pts[np.argmin(diff)] # x-y terkecil → C (60, 400) = Kanan Atas ✅
rect[3] = pts[np.argmax(diff)] # x-y terbesar → B (350, 40) = Kiri Bawah ✅
```

## Kesimpulan

**Penjumlahan (x + y):**
- Menentukan pojok yang paling dekat dan paling jauh dari origin
- Menghasilkan **Kiri Atas** dan **Kanan Bawah**

**Pengurangan (x - y):**
- Membedakan 2 pojok yang tersisa
- Menghasilkan **Kanan Atas** dan **Kiri Bawah**

Dengan 2 perhitungan sederhana ini, kita bisa otomatis mengurutkan 4 pojok tanpa perlu logika kompleks!

