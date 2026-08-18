"""
Demo hasil eksperimen. ALAT DEMO, BUKAN PRODUK.

Tidak memuat model dan tidak menghitung embedding — semua dibaca dari hasil/
yang sudah tersimpan, jadi jalan di laptop tanpa GPU dan tanpa menunggu.

    .venv/bin/pip install streamlit
    cd eksperimen && ../.venv/bin/streamlit run app_demo.py
"""

import glob
import json
import os

import cv2
import numpy as np
import streamlit as st

import protokol as P

AKAR = P.AKAR_HASIL

st.set_page_config(page_title="Preprocessing Re-ID Penyu", layout="wide")

# Warna konsisten di seluruh halaman: hijau = naik, merah = turun,
# abu = selisihnya tidak bisa dibedakan dari nol.
HIJAU, MERAH, ABU = "#1a7f37", "#c0392b", "#6e7781"


def warna(delta, signifikan):
    return ABU if not signifikan else (HIJAU if delta > 0 else MERAH)


def tanda(delta, signifikan):
    return "≈" if not signifikan else ("▲" if delta > 0 else "▼")


# --------------------------------------------------------------- sidebar
def run_tersedia():
    return [os.path.basename(d) for d in sorted(glob.glob(os.path.join(AKAR, "*_*_*")))
            if os.path.exists(os.path.join(d, "statistik.json"))]


runs = run_tersedia()
if not runs:
    st.error("Belum ada hasil. Jalankan `python3 jalankan.py` lalu `statistik.py`.")
    st.stop()

with st.sidebar:
    st.header("Run")
    pilih = st.selectbox(
        "dataset / model / transform", runs,
        index=runs.index("reunion_L_squash") if "reunion_L_squash" in runs else 0,
        help="Tiap run = kombinasi dataset, backbone, dan transform "
             "input. Protokol §3 identik di semua run.")
    ds, model, tf = pilih.split("_")

HASIL = os.path.join(AKAR, pilih)


@st.cache_data(show_spinner=False)
def muat_meta(run):
    d = os.path.join(AKAR, run)

    def b(n):
        p = os.path.join(d, n)
        return json.load(open(p)) if os.path.exists(p) else None

    return (b("statistik.json"), b("header.json"), b("kasus_gagal.json"),
            b("lanjutan.json"), b("grid_rerank.json"))


@st.cache_data(show_spinner=False)
def muat_split(dataset):
    data = os.path.join(P.REPO, "dataset_penyu",
                        {"reunion": "ReunionTurtles",
                         "seaturtleheads": "SeaTurtleIDHeads"}[dataset])
    kat = (P._katalog_reunion if dataset == "reunion"
           else P._katalog_seaturtleheads)(data)
    return P.bangun_split(kat)


@st.cache_data(show_spinner=False)
def muat_emb(run, kondisi, n_gal):
    p = os.path.join(AKAR, run, f"emb_{kondisi}.npy")
    if not os.path.exists(p):
        return None, None
    E = np.load(p)
    return E[:n_gal], E[n_gal:]


def baca_rgb(path):
    im = cv2.imread(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else None


def bingkai(im, benar):
    if not benar:
        return im
    return cv2.copyMakeBorder(im, 10, 10, 10, 10, cv2.BORDER_CONSTANT,
                              value=(26, 127, 55))


stat, header, gagal, lanjut, grid = muat_meta(pilih)
gal, qry = muat_split(ds)

# ------------------------------------------------------------------ judul
st.title("Pengaruh preprocessing terhadap akurasi re-ID penyu")
if header:
    st.caption(
        f"**{header['dataset']}** · hash `{header['dataset_hash']}` · "
        f"**{header['model']}** frozen · transform `{tf}` · "
        f"torch {header['torch']} / timm {header['timm']}")
    s = header["sanity"]
    k = st.columns(4)
    k[0].metric("Query", s["n_query"])
    k[1].metric("Gallery", s["n_gallery"])
    k[2].metric("Individu", s["n_identitas_gallery"])
    k[3].metric("Baseline Rank-1", f"{stat['raw']['rank1']:.1f}%")

if ds == "seaturtleheads":
    st.info("Run ini memakai SeaTurtleIDHeads — fotonya sudah crop kepala dan "
            "tidak punya kolom spesies. Untuk hasil sesuai spesifikasi, pilih "
            "run `reunion_*` di sidebar.", icon="ℹ️")

tab = st.tabs(["📊 Hasil", "🔗 Stage-2 matcher", "🔍 Uji sendiri",
               "🖼 Before / after", "❌ Kasus gagal", "🧪 Cara lain"])

# ================================================================ 1. HASIL
with tab[0]:
    st.subheader("Tabel utama")
    st.caption("Δ dibandingkan raw pada himpunan query yang identik. "
               "**Abu-abu = selisihnya tidak bisa dibedakan dari nol** — itu "
               "bukan 'sedikit lebih buruk', itu 'tidak ada bukti'.")

    baris = []
    for k, b in stat.items():
        d, m, dm = b.get("delta_rank1"), b.get("mcnemar_rank1"), b.get("delta_mAP")
        if d is None:
            baris.append(
                f"<tr><td><b>{b['label']}</b></td>"
                f"<td align=right><b>{b['rank1']:.2f}</b></td>"
                f"<td align=right>{b['rank5']:.2f}</td>"
                f"<td align=right>{b['mAP']:.2f}</td>"
                f"<td align=center>—</td><td align=center>—</td>"
                f"<td align=center>—</td></tr>")
            continue
        sig = m["p_value"] < 0.05
        c, cm = warna(d["delta"], sig), warna(dm["delta"], dm["signifikan"])
        baris.append(
            f"<tr><td>{b['label']}</td>"
            f"<td align=right>{b['rank1']:.2f}</td>"
            f"<td align=right>{b['rank5']:.2f}</td>"
            f"<td align=right>{b['mAP']:.2f}</td>"
            f"<td align=right style='color:{c};white-space:nowrap'>"
            f"{tanda(d['delta'], sig)} {d['delta']:+.2f} "
            f"<span style='opacity:.65'>[{d['ci95'][0]:+.2f}, "
            f"{d['ci95'][1]:+.2f}]</span></td>"
            f"<td align=right style='color:{cm};white-space:nowrap'>"
            f"{tanda(dm['delta'], dm['signifikan'])} {dm['delta']:+.2f}</td>"
            f"<td align=center style='color:{c}'>{m['p_value']:.3g}"
            f"{' ✔' if sig else ''}</td></tr>")

    st.markdown(
        "<table style='width:100%;border-collapse:collapse;font-size:0.92rem'>"
        "<thead><tr style='border-bottom:2px solid #888'>"
        "<th align=left>Kondisi</th><th align=right>Rank-1</th>"
        "<th align=right>Rank-5</th><th align=right>mAP</th>"
        "<th align=right>Δ Rank-1 (95% CI)</th><th align=right>Δ mAP</th>"
        "<th align=center>p</th></tr></thead>"
        f"<tbody>{''.join(baris)}</tbody></table>", unsafe_allow_html=True)

    n = stat["raw"]["n"]
    st.caption(f"n = {n} query · satu prediksi berubah ≈ {100 / n:.2f} poin persen")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Per sisi (Rank-1)")
        st.dataframe([{"Kondisi": b["label"],
                       "Kiri": round(b["per_sisi"]["left"]["rank1"], 2),
                       "Kanan": round(b["per_sisi"]["right"]["rank1"], 2)}
                      for b in stat.values()], hide_index=True, width="stretch")
    with c2:
        st.subheader("Per spesies (Rank-1)")
        sp = sorted(stat["raw"].get("per_spesies", {}))
        if sp:
            st.dataframe(
                [dict({"Kondisi": b["label"]},
                      **{s_: round(b["per_spesies"][s_]["rank1"], 2) for s_ in sp})
                 for b in stat.values()], hide_index=True, width="stretch")
        else:
            st.info("Dataset ini tidak menyimpan spesies.")

    ab = os.path.join(AKAR, f"{ds}_transform_ab.json")
    if os.path.exists(ab):
        with st.expander("Transform input mana yang benar untuk dataset ini?"):
            r = json.load(open(ab))
            k_cfg = [k for k in r if k.startswith(("cfg", "kanonik"))][0]
            k_sq = [k for k in r if k.startswith("squash")][0]
            d = [r[k] for k in r if k.startswith("delta_rank1")][0]
            st.write(
                f"Sesuai `config.json` (bicubic + center crop): Rank-1 "
                f"**{r[k_cfg]['rank1']:.2f}**. Resize langsung ke persegi: "
                f"**{r[k_sq]['rank1']:.2f}**. Δ {d['delta']:+.2f} "
                f"[{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}] — "
                f"{'**signifikan**' if d['signifikan'] else 'tidak signifikan'}.")

# ======================================================== 2. STAGE-2 MATCHER
with tab[1]:
    st.subheader("Stage-2: re-ranking dengan local feature matcher")
    st.caption("Stage 1 (embedding global) mengambil top-k kandidat, stage 2 "
               "mencocokkan sisik satu per satu lalu mengurutkan ulang. "
               "Stage-1 selalu memakai embedding raw — yang divariasikan hanya "
               "gambar yang masuk ke matcher.")

    if not grid:
        st.info("Belum ada. Jalankan `MODEL=L python3 grid_rerank.py` "
                "berulang sampai selesai, lalu `--lapor`.")
    else:
        st.caption(f"k = {grid['k']} · n = {grid['n']} query · "
                   f"model {grid['model']} · dibuat {grid['dibuat']}")

        matchers = sorted({b["matcher"] for b in grid["baris"]} - {"-"})
        c = st.columns([2, 2, 3])
        pilih_m = c[0].multiselect("Matcher", matchers, default=matchers)
        mode_p = c[1].multiselect("Mode gabung", ["murni", "rrf"],
                                  default=["murni", "rrf"],
                                  help="murni = urut hanya dengan skor matcher. "
                                       "rrf = fusi peringkat cosine + matcher, "
                                       "tanpa parameter yang disetel.")
        kond_p = c[2].multiselect(
            "Preprocessing sebelum matching",
            sorted({b["kondisi"] for b in grid["baris"]} - {"-"}),
            default=["raw"])

        baris = []
        for b in grid["baris"]:
            tampil = (b["matcher"] == "-" or
                      (b["matcher"] in pilih_m and b.get("mode") in mode_p
                       and b["kondisi"] in kond_p))
            if not tampil:
                continue
            d = b.get("delta_rank1")
            if d is None:
                baris.append(
                    f"<tr style='background:#00000010'>"
                    f"<td><b>{b['label']}</b></td>"
                    f"<td align=right><b>{b['rank1']:.2f}</b></td>"
                    f"<td align=right>{b['rank5']:.2f}</td>"
                    f"<td align=right><b>{b['mAP']:.2f}</b></td>"
                    f"<td align=right>{b['hijau']:.2f}</td>"
                    f"<td align=right>{b['sisik']:.2f}</td>"
                    f"<td align=center>—</td><td align=center>—</td></tr>")
                continue
            sig = b["mcnemar_rank1"]["p_value"] < 0.05
            col = warna(d["delta"], sig)
            baris.append(
                f"<tr><td>{b['label']}</td>"
                f"<td align=right>{b['rank1']:.2f}</td>"
                f"<td align=right>{b['rank5']:.2f}</td>"
                f"<td align=right>{b['mAP']:.2f}</td>"
                f"<td align=right>{b['hijau']:.2f}</td>"
                f"<td align=right>{b['sisik']:.2f}</td>"
                f"<td align=right style='color:{col};white-space:nowrap'>"
                f"{tanda(d['delta'], sig)} {d['delta']:+.2f} "
                f"<span style='opacity:.65'>[{d['ci95'][0]:+.2f}, "
                f"{d['ci95'][1]:+.2f}]</span></td>"
                f"<td align=center style='color:{col}'>"
                f"{b['mcnemar_rank1']['p_value']:.3g}{' ✔' if sig else ''}</td></tr>")

        st.markdown(
            "<table style='width:100%;border-collapse:collapse;font-size:0.9rem'>"
            "<thead><tr style='border-bottom:2px solid #888'>"
            "<th align=left>Konfigurasi</th><th align=right>Rank-1</th>"
            "<th align=right>Rank-5</th><th align=right>mAP</th>"
            "<th align=right>hijau</th><th align=right>sisik</th>"
            "<th align=right>Δ Rank-1 (95% CI)</th><th align=center>p</th>"
            f"</tr></thead><tbody>{''.join(baris)}</tbody></table>",
            unsafe_allow_html=True)

        st.divider()
        st.markdown(
            "**Kenapa XFeat menang telak dan SIFT tidak.** SIFT mencocokkan "
            "tekstur apa saja yang konsisten secara geometris — termasuk "
            "karang dan pasir di latar, yang antar foto di lokasi sama "
            "menghasilkan inlier melimpah. Diukur: inlier pasangan **salah** "
            "mengalahkan inlier pasangan **benar** di 80% query. XFeat "
            "terlatih untuk korespondensi yang benar, dan efeknya paling "
            "terlihat di penyu sisik: SIFT menjatuhkannya dari 29% ke 6%, "
            "XFeat menaikkannya ke 43%.")

        if grid.get("belum_ada_bobot"):
            st.warning(
                "**Belum bisa dijalankan — angkanya sengaja TIDAK dikarang:**\n\n"
                + "\n".join(f"- **{x['matcher']}** — {x['alasan']}"
                            for x in grid["belum_ada_bobot"])
                + "\n\nJalankan `unduh_matcher.py` di Mac untuk mengisinya "
                  "(cara yang sama sudah berhasil untuk XFeat).", icon="⚠️")


# =========================================================== 3. UJI SENDIRI
with tab[2]:
    st.subheader("Uji satu foto pada beberapa kondisi sekaligus")
    st.caption("Pilih satu foto query, lihat top-k gallery-nya berubah — atau "
               "tidak berubah — antar kondisi. Bingkai hijau = jawaban benar.")

    c = st.columns([3, 1, 1])
    ind = c[0].selectbox("Individu", sorted({r["identity"] for r in qry}))
    sisi_ada = sorted({r["side"] for r in qry if r["identity"] == ind})
    sisi = c[1].selectbox("Sisi", sisi_ada)
    topk = c[2].slider("Top-k", 3, 10, 5)

    kandidat = [i for i, r in enumerate(qry)
                if r["identity"] == ind and r["side"] == sisi]
    if not kandidat:
        st.warning("Tidak ada foto query untuk kombinasi itu.")
    else:
        qi = kandidat[0]
        q = qry[qi]
        id_g = np.array([r["identity"] for r in gal])
        s_g = np.array([r["side"] for r in gal])
        n_sisi = int((s_g == q["side"]).sum())

        cc = st.columns([1, 4])
        qimg = baca_rgb(q["path"])
        if qimg is not None:
            cc[0].image(qimg, width="stretch",
                        caption=f"QUERY · {q['side']} · {q['year']}")
        cc[1].markdown(
            f"Dicari di **{n_sisi}** foto gallery sisi `{q['side']}` "
            f"(sisi lain dikunci, tidak boleh dicocokkan silang). "
            f"Tebak acak = **{100 / max(n_sisi, 1):.2f}%**.")

        pilih_kondisi = st.multiselect(
            "Kondisi yang dibandingkan", list(P.KONDISI),
            default=[k for k in ("raw", "wb", "gray") if k in P.KONDISI],
            format_func=lambda k: P.LABEL[k])

        for k in pilih_kondisi:
            Eg, Eq = muat_emb(pilih, k, len(gal))
            if Eg is None:
                st.warning(f"{P.LABEL[k]}: embedding belum ada di run ini.")
                continue
            s = Eq[qi] @ Eg.T
            s = np.where(s_g == q["side"], s, -np.inf)
            urut_penuh = np.argsort(-s)
            rank_benar = int(np.flatnonzero(
                id_g[urut_penuh] == q["identity"])[0]) + 1
            ok = rank_benar == 1
            st.markdown(
                f"**{P.LABEL[k]}** — <span style='color:"
                f"{HIJAU if ok else MERAH}'>"
                f"{'BENAR di rank-1' if ok else f'salah — jawaban benar di rank {rank_benar}'}"
                f"</span>", unsafe_allow_html=True)
            kol = st.columns(topk)
            for n_, j in enumerate(urut_penuh[:topk]):
                im = baca_rgb(gal[j]["path"])
                if im is None:
                    continue
                benar = id_g[j] == q["identity"]
                kol[n_].image(
                    bingkai(im, benar), width="stretch",
                    caption=f"#{n_ + 1} {gal[j]['identity'].split('/')[-1]} "
                            f"{gal[j]['year']}\n{s[j]:.3f}"
                            + ("  ✅" if benar else ""))
            st.divider()

# ========================================================= 3. BEFORE/AFTER
with tab[3]:
    st.subheader("Apa yang sebenarnya dilakukan tiap preprocessing")
    c = st.columns(2)
    ind2 = c[0].selectbox("Individu", sorted({r["identity"] for r in qry}),
                          key="ba_ind")
    kand = [r for r in qry if r["identity"] == ind2]
    r = kand[c[1].selectbox(
        "Foto", range(len(kand)), key="ba_foto",
        format_func=lambda i: f"{kand[i]['side']} · {kand[i]['year']}")]
    rgb = baca_rgb(r["path"])
    if rgb is None:
        st.warning("Gambar tidak terbaca.")
    else:
        kols = st.columns(3)
        for i, k in enumerate(P.KONDISI):
            kols[i % 3].image(P.KONDISI[k](rgb), caption=P.LABEL[k],
                              width="stretch")

# ========================================================== 4. KASUS GAGAL
with tab[4]:
    if not gagal:
        st.info("Belum ada. Jalankan `python3 kasus_gagal.py`.")
    else:
        st.write(f"**{gagal['total_gagal']} dari {gagal['total_query']}** query "
                 f"salah di rank-1 pada kondisi `{gagal['kondisi']}`.")
        DUGAAN = {
            "yakin_tapi_salah":
                "Skor top-1 tinggi tapi identitasnya salah — embedding menangkap "
                "kemiripan pose dan pencahayaan, bukan pola sisik. Mode kegagalan "
                "paling berbahaya: sistem terlihat yakin.",
            "jawaban_jauh":
                "Jawaban benar terlempar sangat jauh — sudut atau jarak foto beda "
                "ekstrem. Preprocessing tidak akan menolong kasus ini.",
            "nyaris_peringkat_2":
                "Jawaban benar di peringkat 2 — tertolong verifikasi manusia. "
                "Inilah alasan Rank-5 tetap dilaporkan.",
        }
        for i, k in enumerate(gagal["contoh"], 1):
            st.markdown(f"#### {i}. `{k['query_id']}` — {k['jenis']}")
            st.caption(DUGAAN.get(k["jenis"], ""))
            kol = st.columns(len(k["top5"]) + 1)
            q = baca_rgb(os.path.join(P.REPO, k["query_path"]))
            if q is not None:
                kol[0].image(q, caption=f"QUERY {k['query_year']}", width="stretch")
            for j, g in enumerate(k["top5"], 1):
                im = baca_rgb(os.path.join(P.REPO, g["path"]))
                if im is None:
                    continue
                kol[j].image(
                    bingkai(im, g["benar"]), width="stretch",
                    caption=f"#{j} {g['id'].split('/')[-1]} {g['year']}\n"
                            f"{g['skor']:.3f}" + ("  ✅" if g["benar"] else ""))
            st.caption(f"peringkat jawaban benar: {k['peringkat_jawaban_benar']}")
            st.divider()

# ============================================================= 5. CARA LAIN
with tab[5]:
    st.subheader("Di luar preprocessing")
    if not lanjut:
        st.info("Belum ada. Jalankan `python3 lanjutan.py`.")
    else:
        st.caption("Semua varian memakai embedding yang sudah dihitung — nol "
                   "forward pass tambahan.")
        baris = []
        for nama, b in lanjut.items():
            d = b.get("delta_rank1")
            if d is None:
                baris.append(
                    f"<tr><td><b>{nama}</b></td>"
                    f"<td align=right><b>{b['rank1']:.2f}</b></td>"
                    f"<td align=right>{b['rank5']:.2f}</td>"
                    f"<td align=right>{b['mAP']:.2f}</td>"
                    f"<td align=center>—</td><td align=center>—</td></tr>")
                continue
            sig = b["mcnemar_rank1"]["p_value"] < 0.05
            c = warna(d["delta"], sig)
            baris.append(
                f"<tr><td>{nama}</td>"
                f"<td align=right>{b['rank1']:.2f}</td>"
                f"<td align=right>{b['rank5']:.2f}</td>"
                f"<td align=right>{b['mAP']:.2f}</td>"
                f"<td align=right style='color:{c};white-space:nowrap'>"
                f"{tanda(d['delta'], sig)} {d['delta']:+.2f} "
                f"<span style='opacity:.65'>[{d['ci95'][0]:+.2f}, "
                f"{d['ci95'][1]:+.2f}]</span></td>"
                f"<td align=center style='color:{c}'>"
                f"{b['mcnemar_rank1']['p_value']:.3g}{' ✔' if sig else ''}</td></tr>")
        st.markdown(
            "<table style='width:100%;border-collapse:collapse;font-size:0.92rem'>"
            "<thead><tr style='border-bottom:2px solid #888'>"
            "<th align=left>Varian</th><th align=right>Rank-1</th>"
            "<th align=right>Rank-5</th><th align=right>mAP</th>"
            "<th align=right>Δ Rank-1 (95% CI)</th><th align=center>p</th>"
            f"</tr></thead><tbody>{''.join(baris)}</tbody></table>",
            unsafe_allow_html=True)
        st.warning(
            "**Konsensus dua sisi memakai DUA foto per penyu**, varian lain "
            "memakai satu. Kenaikannya bukan perbaikan algoritma yang gratis — "
            "ia menukar usaha di lapangan dengan akurasi. Konsekuensi UX itu "
            "harus ikut dilaporkan.", icon="⚠️")
