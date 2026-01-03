import cv2
import numpy as np
import os
import pygame 

from game_logic import HigherLowerGame 


pygame.init()
screen = None
clock = pygame.time.Clock()
running = True

WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

FONT_XL = pygame.font.Font(None, 60) 
FONT_LG = pygame.font.Font(None, 40) # Font
FONT_MD = pygame.font.Font(None, 30)
FONT_SM = pygame.font.Font(None, 24) 
# ---

def titik(pts): # urutin matrix yang bener nemuin pojok kiri dl
    rect = np.zeros((4,2), dtype = "float32")
    s = pts.sum(axis = 1) # x+y
    diff = np.diff(pts, axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def klasifikasi_kartu(kartu_warped, all_templates):
    gray_warped = cv2.cvtColor(kartu_warped, cv2.COLOR_BGR2GRAY)
    _, threshold_warped = cv2.threshold(gray_warped, 150, 255,
                                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_match = ("unknow", float('inf'))

    for nama_kartu, gambar in all_templates.items():

        if threshold_warped.shape != gambar.shape:
            gambar = cv2.resize(gambar,
                                (threshold_warped.shape[1], threshold_warped.shape[0]))

        # Sum of A
        img1 = threshold_warped.astype(np.float32)
        img2 = gambar.astype(np.float32)

        diff = np.abs(img1 - img2)
        sad_value = np.sum(diff)

        if sad_value < best_match[1]: # ini cari sad terkecil dair semua templatenya, taruh di array 1
            best_match = (nama_kartu, sad_value)

    kartu_terklasifikasi = best_match[0]
    sad_value = best_match[1]       #
    max_sad = threshold_warped.shape[0] * threshold_warped.shape[1] * 255

    confidence = sad_value / max_sad  # format confidence
    confidence = max(0.0, min(1.0, confidence))  # clamp 0–1

    if confidence > 0.5:
        kartu_terklasifikasi = "tidak diketahui"

    return kartu_terklasifikasi, confidence


Folder = "/Users/gungfebrian/Documents/Tugas/PCV/Templatekartu"

RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUITS = ["Spade","Heart","Diamond","Club"]

kartu_template = {}
lebar, tinggi = 300, 420 # warp

print(f"Nama folder: {Folder}")

for suit in SUITS: # Nested Loop
    for rank in RANKS:
        nama_kartu = f"{rank} {suit}" # contoh output 'K diamond'
        namafile = f"{rank}_{suit}.jpg" # K diamond.jpg
        path = os.path.join(Folder, namafile)

        template = cv2.imread(path)
        if template is None:
            print(f"error {namafile} gaada di {Folder}")
            continue # bassicly loop supaya gampang

        Resized = cv2.resize(template, (lebar, tinggi))
        template_gray = cv2.cvtColor(Resized, cv2.COLOR_BGR2GRAY)

        _, template_thresh = cv2.threshold(template_gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kartu_template[nama_kartu] = template_thresh  # save di ram

print(f"total: {len(kartu_template)}")

# Kamera time
url = "http://10.4.136.188:4747/video"
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("kamera ga bisa")

lebarkamerakartu = 300 # setelh warp
tinggikamerakartu = 420

dst_pts = np.array([ # kalo kartunya miring titik tujuan warp
    [0, 0],
    [lebarkamerakartu -1, 0],
    [lebarkamerakartu - 1, tinggikamerakartu - 1],
    [0, tinggikamerakartu - 1]], dtype="float32")

game = HigherLowerGame() # instansiasi game

# Definisikan lebar panel informasi (seperlima dari lebar kamera)
INFO_PANEL_WIDTH = 300 
CAMERA_OFFSET_X = INFO_PANEL_WIDTH

while running:
    ret, frame = cap.read()
    if not ret:
        print('kamera errorr')
        break

    img_output = frame.copy()
    img_tinggi, img_lebar = frame.shape [:2] # ambil tinggi lebar

    # set up si pygm
    if screen is None:
        # Lebar total: Panel Info + Lebar Kamera
        SCREEN_WIDTH = img_lebar + INFO_PANEL_WIDTH
        SCREEN_HEIGHT = img_tinggi 
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Deteksi Kartu - Kauay Adu Nilai")

    scale_down = 0.5 # supaya lebih cepet detect
    if scale_down != 1.0:
        resized_frame = cv2.resize(frame,
        (int(img_lebar * scale_down), int(img_tinggi * scale_down)),
        interpolation=cv2.INTER_AREA)
    else:
        resized_frame = frame.copy()

    imgbaru_tinggi, imgbaru_lebar = resized_frame.shape[:2]

    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gambar_contour = resized_frame.copy()

    area_kartumin = (imgbaru_lebar * imgbaru_tinggi) / 150 # area minimum untuk filter kontur

    kartu_terdeteksi = [] # list kosong sipan kartu terdetect

    for c_resized in contours:
        area = cv2.contourArea(c_resized)

        if area > area_kartumin:
            peri = cv2.arcLength(c_resized, True)
            approx_resized = cv2.approxPolyDP(c_resized, 0.02 * peri, True)

            if len(approx_resized) == 4:
                approx_original = (approx_resized / scale_down).astype(np.int32)
                src_pts = titik(approx_original.reshape(4,2))


                m = cv2.getPerspectiveTransform(src_pts, dst_pts) # warp
                w_kartu = cv2.warpPerspective(frame, m, (lebar, tinggi))

                # klasifikasi 52 template
                klasifikasi_teks, confidence = klasifikasi_kartu(w_kartu, kartu_template)

                # simpan info kartu
                kartu_terdeteksi.append({
                    'contour': approx_original,
                    'klasifikasi': klasifikasi_teks,
                    'confidence': confidence
                })

    # --- LOGIKA INTEGRASI CV & GAME BARU ---
    kartu_valid = None
    if len(kartu_terdeteksi) > 0:
        # Hanya ambil satu kartu dengan confidence terbaik (terkecil)
        best_card = min(kartu_terdeteksi, key=lambda x: x['confidence'])
        BATAS_CONFIDENCE = 0.3

        if best_card['confidence'] < BATAS_CONFIDENCE:
            kartu_valid = best_card['klasifikasi'] # Contoh: "K Diamond"

        # Panggil fungsi proses kartu di Game Logic
        if kartu_valid:
            # Menggunakan metode set_card yang diimplementasikan di game_logic.py
            game.set_card_from_cv(kartu_valid)
        else:
            # Tetap gunakan CV2 untuk feedback contour di frame
            cv2.putText(img_output, "Confidence Rendah!", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    for info_kartu in kartu_terdeteksi: # loop
        contour = info_kartu['contour']
        klasifikasi_teks = info_kartu['klasifikasi']
        confidence = info_kartu['confidence']

        warna_contour = (0,255,0) if confidence < 0.3 else (0,255,255) # hijau tinggi kuning kecil

        cv2.drawContours(img_output, [contour], -1, warna_contour, 2) # gambar kotak contour

        m_cnt = cv2.moments(contour) # titik tengah contour untuk ngasi teks
        if m_cnt['m00'] != 0:
            cX = int(m_cnt['m10'] / m_cnt['m00'])
            cY = int(m_cnt['m01'] / m_cnt['m00'])

            cv2.putText(img_output, klasifikasi_teks, (cX - 50, cY),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255, 0), 2) # nama kartu di frame

            cv2.putText(img_output, f'confidence: {confidence:.2f}',
                (cX - 50, cY +20),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5, (0, 255, 0), 1)

    # --- AMBIL NILAI UNTUK PYGAME UI ---
    nilai_acak_murni = game.get_value(game.current_card) if game.current_card else 'N/A'

    if game.reveal_current_card:
        display_acuan = game.current_card
        display_acuan_value = nilai_acak_murni
    else:
        display_acuan = "???"
        display_acuan_value = "?"
        
    nilai_tebakan = game.get_value(game.next_card) if game.next_card else 'N/A'
    
    # --- PYGAME RENDERING START ---
    
    # 1. Konversi frame OpenCV ke Pygame
    rgb_frame = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)
    rgb_frame = np.rot90(rgb_frame)
    surface = pygame.surfarray.make_surface(rgb_frame)
    surface = pygame.transform.flip(surface, True, False) # Flip frame jika perlu

    # 2. Bersihkan layar dan gambar Surface di sebelah KANAN
    screen.fill(BLACK) # Isi latar belakang hitam
    screen.blit(surface, (CAMERA_OFFSET_X, 0)) # Gambar kamera di offset X=INFO_PANEL_WIDTH
    
    # 3. RENDER GAME UI DI PANEL KIRI
    
    y_pos = 20
    
    # JUDUL
    text_title = FONT_LG.render("Tebak Nilai Kartu", True, YELLOW)
    screen.blit(text_title, (10, y_pos))
    y_pos += 50
    
    # SKOR
    text_score = FONT_XL.render(f"SCORE: {game.score}", True, GREEN)
    screen.blit(text_score, (10, y_pos))
    y_pos += 80

    text_acuan_label = FONT_MD.render("KARTU Adalah:", True, WHITE)
    screen.blit(text_acuan_label, (10, y_pos))
    y_pos += 30
    
    text_acuan_card = FONT_LG.render(f"{display_acuan}", True, WHITE)
    screen.blit(text_acuan_card, (10, y_pos))
    y_pos += 40
    
    text_acuan_value = FONT_LG.render(f"VALUE: {display_acuan_value}", True, (0, 255, 255))
    screen.blit(text_acuan_value, (10, y_pos))
    y_pos += 70

    if game.next_card or "Tekan 'N'" in game.message:
        text_tebakan_label = FONT_MD.render("KARTU TEBAKAN:", True, BLUE)
        screen.blit(text_tebakan_label, (10, y_pos))
        y_pos += 30

        if game.next_card:
            text_tebakan_card = FONT_LG.render(f"{game.next_card}", True, BLUE)
            text_tebakan_value = FONT_LG.render(f"VALUE: {nilai_tebakan}", True, (0, 255, 255))
            screen.blit(text_tebakan_card, (10, y_pos))
            y_pos += 40
            screen.blit(text_tebakan_value, (10, y_pos))
            y_pos += 50

    if "Tekan 'N'" in game.message or "MENANG" in game.message:
  
        if "SALAH" in game.message or "❌" in game.message:
            display_text = "Salahhh"
            color = RED
        elif "BENAR" in game.message or "✅" in game.message:
            display_text = "✅ Benarrrr"
            color = GREEN
        elif "MENANG" in game.message:
            display_text = "Yeyy Menangg"
            color = YELLOW

        text_result_xl = FONT_XL.render(display_text, True, color)
        
        screen.blit(text_result_xl, (
            INFO_PANEL_WIDTH // 2 - text_result_xl.get_width() // 2, 
            450 
        ))
    
  
    if "SALAH" in game.message or "Wrong" in game.message or "❌" in game.message:
        color = RED
    elif "BENAR" in game.message or "Correct" in game.message or "MENANG" in game.message or "✅" in game.message:
        color = GREEN
    else:
        color = WHITE
    


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_r: 
                game.start_game()
            elif event.key == pygame.K_n: # next round n
                game.start_new_round_card()
            elif event.key == pygame.K_h:
                game.check_guess("higher")
            elif event.key == pygame.K_l:
                game.check_guess("lower")
    
    pygame.display.flip()
    clock.tick(60)


cap.release()
cv2.destroyAllWindows()
pygame.quit()