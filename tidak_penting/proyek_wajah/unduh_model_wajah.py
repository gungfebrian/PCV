"""
Unduh model deteksi & pengenalan wajah dari OpenCV Zoo ke models_wajah/.

    yunet.onnx   ~230 KB  deteksi wajah + 5 landmark
    sface.onnx   ~37 MB   embedding wajah 128 dimensi

Jalankan:
    .venv/bin/python unduh_model_wajah.py
"""

import os
import urllib.error
import urllib.request

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_wajah")
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

BERKAS = [
    ("yunet.onnx",
     f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    ("sface.onnx",
     f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
]


def unduh():
    os.makedirs(DIR, exist_ok=True)
    for nama, url in BERKAS:
        tujuan = os.path.join(DIR, nama)
        if os.path.exists(tujuan) and os.path.getsize(tujuan) > 1000:
            print(f"  {nama} sudah ada ({os.path.getsize(tujuan)/1e6:.1f} MB)")
            continue
        print(f"  mengunduh {nama} ...")
        try:
            urllib.request.urlretrieve(url, tujuan)
            print(f"  ok {nama} ({os.path.getsize(tujuan)/1e6:.1f} MB)")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  GAGAL {nama}: {e}")
            return False
    return True


if __name__ == "__main__":
    print(f"Model wajah -> {DIR}")
    print("Selesai." if unduh() else "Ada yang gagal diunduh.")
