"""
Demo tipis hasil eksperimen. INI ALAT DEMO, BUKAN PRODUK.

Tidak memuat model dan tidak menghitung embedding — semuanya dibaca dari
hasil/ yang sudah tersimpan. Jadi bisa jalan di laptop tanpa GPU dan tanpa
menunggu.

    .venv/bin/pip install streamlit
    .venv/bin/streamlit run eksperimen/app_demo.py
"""

import json
import os

import cv2
import numpy as np
import streamlit as st

import protokol as P

HASIL = os.path.join(P.BASE, "hasil", f"{P.DATASET}_{P.MODEL}_{P.TRANSFORM}")

st.set_page_config(page_title="Preprocessing Re-ID Penyu", layout="wide")


@st.cache_data
def muat_semua():
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    stat = json.load(open(os.path.join(HASIL, "statistik.json")))
    header = json.load(open(os.path.join(HASIL, "header.json")))
    gagal = json.load(open(os.path.join(HASIL, "kasus_gagal.json")))
    return gal, qry, stat, header, gagal


@st.cache_data
def muat_emb(nama, n_gal):
    E = np.load(os.path.join(HASIL, f"emb_{nama}.npy"))
    return E[:n_gal], E[n_gal:]


def baca_rgb(rel):
    im = cv2.imread(os.path.join(P.REPO, rel))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else None


gal, qry, stat, header, gagal = muat_semua()

st.title("Pengaruh preprocessing terhadap akurasi re-ID penyu")
st.caption(
    f"Dataset **{header['dataset']}** (hash `{header['dataset_hash']}`) · "
    f"model **{header['model']}** frozen · transform `{P.TRANSFORM}` · "
    f"torch {header['torch']} / timm {header['timm']}"
)
st.warning(
    "Spesifikasi meminta ReunionTurtles. Dataset itu tidak tersedia di "
    "lingkungan ini, jadi protokol dijalankan di SeaTurtleIDHeads. "
    "Konsekuensi: tidak ada breakdown per spesies, dan foto sudah berupa crop "
    "kepala sehingga kondisi 'crop kepala' berarti crop yang LEBIH ketat.",
    icon="⚠️")

tab1, tab2, tab3 = st.tabs(["Tabel hasil", "Before / after", "Kasus gagal"])

# ------------------------------------------------------------- tabel
with tab1:
    baris = []
    for k, b in stat.items():
        d = b.get("delta_rank1")
        m = b.get("mcnemar_rank1")
        dm = b.get("delta_mAP")
        baris.append({
            "Kondisi": b["label"],
            "Rank-1": round(b["rank1"], 2),
            "Rank-5": round(b["rank5"], 2),
            "mAP": round(b["mAP"], 2),
            "Δ Rank-1 vs raw (95% CI)": "—" if not d else
                f"{d['delta']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]",
            "p (McNemar)": "—" if not m else f"{m['p_value']:.3g}",
            "Δ mAP (95% CI)": "—" if not dm else
                f"{dm['delta']:+.2f} [{dm['ci95'][0]:+.2f}, {dm['ci95'][1]:+.2f}]",
        })
    st.dataframe(baris, width='stretch', hide_index=True)
    st.caption(f"n = {stat['raw']['n']} query, gallery {header['sanity']['n_gallery']} "
               f"foto / {header['sanity']['n_identitas_gallery']} individu "
               f"({header['sanity']['n_identitas_gallery'] - header['sanity']['n_identitas_query']} "
               f"di antaranya distraktor tanpa jawaban benar).")

    st.subheader("Breakdown per sisi (Rank-1)")
    st.dataframe([{"Kondisi": b["label"],
                   "Kiri": round(b["per_sisi"]["left"]["rank1"], 2),
                   "Kanan": round(b["per_sisi"]["right"]["rank1"], 2)}
                  for b in stat.values()],
                 width='stretch', hide_index=True)
    st.caption("Kolom spesies tidak ada: SeaTurtleIDHeads tidak menyimpan spesies.")

    ab = os.path.join(HASIL, "..", "transform_ab.json")
    if os.path.exists(ab):
        with st.expander("Catatan: transform input mana yang benar?"):
            r = json.load(open(ab))
            st.write(
                f"`crop_pct 0.9 + bicubic` sesuai config.json memberi Rank-1 "
                f"**{r['kanonik_bicubic_crop0.9']['rank1']:.2f}**, sedangkan "
                f"resize langsung 224x224 memberi "
                f"**{r['squash_INTER_AREA_224']['rank1']:.2f}** "
                f"(Δ {r['delta_rank1_squash_minus_kanonik']['delta']:+.2f}, "
                f"McNemar p={r['mcnemar']['p_value']:.2g}). "
                "Center crop membuang sisik di tepi pada foto yang sudah "
                "berupa crop kepala.")

# -------------------------------------------------------- before/after
with tab2:
    kondisi = st.selectbox("Kondisi", list(P.KONDISI),
                           format_func=lambda k: P.LABEL[k], index=1)
    idx = st.slider("Foto query ke-", 0, len(qry) - 1, 0)
    r = qry[idx]
    rgb = baca_rgb(os.path.relpath(r["path"], P.REPO))
    c1, c2 = st.columns(2)
    if rgb is not None:
        c1.image(rgb, caption=f"raw — {r['identity']} / {r['side']} / {r['year']}",
                 width='stretch')
        c2.image(P.KONDISI[kondisi](rgb), caption=P.LABEL[kondisi],
                 width='stretch')

# --------------------------------------------------------- kasus gagal
with tab3:
    st.write(f"**{gagal['total_gagal']} dari {gagal['total_query']}** query salah "
             f"di rank-1 pada kondisi `{gagal['kondisi']}`.")
    DUGAAN = {
        "yakin_tapi_salah":
            "Skor top-1 tinggi tapi identitasnya salah — embedding menangkap "
            "kemiripan pose/pencahayaan, bukan pola sisik.",
        "jawaban_jauh":
            "Jawaban benar terlempar sangat jauh — foto gallery dan query "
            "kemungkinan beda sudut/jarak ekstrem, atau kualitas foto buruk.",
        "nyaris_peringkat_2":
            "Jawaban benar di peringkat 2 — di praktik konservasi kasus ini "
            "tertolong verifikasi manusia, karena itu Rank-5 tetap dilaporkan.",
    }
    for i, k in enumerate(gagal["contoh"], 1):
        st.markdown(f"### {i}. `{k['query_id']}` — {k['jenis']}")
        st.caption(DUGAAN.get(k["jenis"], ""))
        kol = st.columns(6)
        q = baca_rgb(k["query_path"])
        if q is not None:
            kol[0].image(q, caption=f"QUERY {k['query_id']}\n{k['query_year']}",
                         width='stretch')
        for j, g in enumerate(k["top5"], 1):
            im = baca_rgb(g["path"])
            if im is not None:
                kol[j].image(im, width='stretch',
                             caption=f"#{j} {g['id']} {g['year']}\n"
                                     f"{g['skor']:.3f}"
                                     + ("  ✅" if g["benar"] else ""))
        st.caption(f"peringkat jawaban benar: {k['peringkat_jawaban_benar']}")
        st.divider()
