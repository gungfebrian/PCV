if __name__ == "__main__":
    # Simple CLI for Demo
    game = HigherLowerGame()
    print("="*40)
    print("   HIGHER OR LOWER - CONSOLE DEMO")
    print("="*40)
    
    while not game.is_game_over:
        print(f"\n[SCORE]: {game.score}")
        print(f"[CARD] : {game.current_card}")
        print("-" * 20)
        
        choice = input("Next card Higher (h) or Lower (l)? ").lower().strip()
        
        if choice not in ['h', 'l', 'higher', 'lower']:
            print("Invalid input! Type 'h' or 'l'.")
            continue
            
        guess = "higher" if choice in ['h', 'higher'] else "lower"
        result = game.check_guess(guess)
        
        print(f"\n>>> {game.message}")
        if not result:
            # Optional: End game on wrong guess for demo thrill
            # game.is_game_over = True
            pass
            
    print("\nGAME OVER")
    print(f"Final Score: {game.score}")import tkinter as tk
from tkinter import messagebox
import random

# ==============================================================================
# 1. LOGIKA GAME (OTAK) - SAMA PERSIS DENGAN YANG SUDAH KITA BUAT
# ==============================================================================
class GameTebakKartu:
    def __init__(self):
        self.daftar_angka = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        self.jenis_bunga  = ["Spade", "Heart", "Diamond", "Club"]
        
        self.kamus_nilai = {
            "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, 
            "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13
        }
        
        # Status Game
        self.skor = 0
        self.game_selesai = False
        self.tumpukan_kartu = []
        self.kartu_sekarang = None
        self.kartu_selanjutnya = None
        self.pesan_status = "Selamat Datang!"
        
        self.mulai_game_baru()

    def buat_tumpukan_baru(self):
        tumpukan_baru = []
        for bunga in self.jenis_bunga:
            for angka in self.daftar_angka:
                tumpukan_baru.append(f"{angka} {bunga}")
        return tumpukan_baru

    def mulai_game_baru(self):
        self.skor = 0
        self.game_selesai = False
        self.tumpukan_kartu = self.buat_tumpukan_baru()
        random.shuffle(self.tumpukan_kartu)
        
        if len(self.tumpukan_kartu) > 0:
            self.kartu_sekarang = self.tumpukan_kartu.pop()
        else:
            self.game_selesai = True

    def terjemahkan_ke_angka(self, nama_kartu):
        if not nama_kartu: return 0
        pecahan_kata = nama_kartu.split(" ")
        return self.kamus_nilai.get(pecahan_kata[0], 0)

    def cek_tebakan(self, pilihan_pemain):
        if self.game_selesai: return False

        if len(self.tumpukan_kartu) == 0:
            self.game_selesai = True
            self.pesan_status = "Menang! Deck Habis."
            return True

        self.kartu_selanjutnya = self.tumpukan_kartu.pop()
        
        nilai_lama = self.terjemahkan_ke_angka(self.kartu_sekarang)
        nilai_baru = self.terjemahkan_ke_angka(self.kartu_selanjutnya)
        jawaban_benar = False
    
        if pilihan_pemain == "tinggi" and nilai_baru > nilai_lama:
            jawaban_benar = True   
        elif pilihan_pemain == "rendah" and nilai_baru < nilai_lama:
            jawaban_benar = True

        if jawaban_benar:
            self.skor += 1
            self.pesan_status = "BENAR! Lanjut..."
        else:
            self.skor = max(0, self.skor - 1)
            self.pesan_status = f"SALAH! Munculnya {self.kartu_selanjutnya}"
            
        self.kartu_sekarang = self.kartu_selanjutnya
        return jawaban_benar

# ==============================================================================
# 2. UI / TAMPILAN (TKINTER) - WAJAH APLIKASI
# ==============================================================================
class GameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Higher or Lower - Comp. Eng ITS")
        self.root.geometry("400x600")
        self.root.configure(bg="#2c3e50") # Warna background elegan

        # Inisialisasi Logika Game
        self.game = GameTebakKartu()

        # --- HEADER (JUDUL & SKOR) ---
        self.header_frame = tk.Frame(root, bg="#2c3e50")
        self.header_frame.pack(pady=20)

        self.lbl_title = tk.Label(self.header_frame, text="HIGHER OR LOWER", font=("Helvetica", 18, "bold"), fg="white", bg="#2c3e50")
        self.lbl_title.pack()

        self.lbl_score = tk.Label(self.header_frame, text=f"Score: {self.game.skor}", font=("Helvetica", 14), fg="#f1c40f", bg="#2c3e50")
        self.lbl_score.pack()

        # --- AREA KARTU (CARD DISPLAY) ---
        # Kita bikin kotak putih biar mirip kartu asli
        self.card_frame = tk.Frame(root, bg="white", width=200, height=280, relief="raised", bd=5)
        self.card_frame.pack(pady=20)
        self.card_frame.pack_propagate(False) # Agar ukuran frame tidak menyusut

        # Simbol Kartu (Tengah Besar)
        self.lbl_card_symbol = tk.Label(self.card_frame, text="♦", font=("Times New Roman", 100), bg="white", fg="red")
        self.lbl_card_symbol.place(relx=0.5, rely=0.5, anchor="center")

        # Teks Kartu (Pojok Kiri Atas)
        self.lbl_card_text_top = tk.Label(self.card_frame, text="10", font=("Helvetica", 20, "bold"), bg="white", fg="red")
        self.lbl_card_text_top.place(x=10, y=10)

        # Teks Kartu (Pojok Kanan Bawah - Terbalik)
        self.lbl_card_text_bottom = tk.Label(self.card_frame, text="10", font=("Helvetica", 20, "bold"), bg="white", fg="red")
        self.lbl_card_text_bottom.place(x=150, y=230)

        # --- PESAN STATUS ---
        self.lbl_status = tk.Label(root, text=self.game.pesan_status, font=("Helvetica", 12), fg="white", bg="#2c3e50")
        self.lbl_status.pack(pady=10)

        # --- TOMBOL KONTROL ---
        self.btn_frame = tk.Frame(root, bg="#2c3e50")
        self.btn_frame.pack(pady=20)

        self.btn_higher = tk.Button(self.btn_frame, text="▲ HIGHER", font=("Helvetica", 12, "bold"), bg="#27ae60", fg="white", width=12, command=lambda: self.proses_tebakan("tinggi"))
        self.btn_higher.grid(row=0, column=0, padx=10)

        self.btn_lower = tk.Button(self.btn_frame, text="▼ LOWER", font=("Helvetica", 12, "bold"), bg="#c0392b", fg="white", width=12, command=lambda: self.proses_tebakan("rendah"))
        self.btn_lower.grid(row=0, column=1, padx=10)

        # --- TOMBOL DEMO & RESET ---
        self.control_frame = tk.Frame(root, bg="#2c3e50")
        self.control_frame.pack(pady=10)

        self.btn_reset = tk.Button(self.control_frame, text="Reset Game", command=self.reset_game, width=15)
        self.btn_reset.grid(row=0, column=0, padx=5)
        
        self.btn_demo = tk.Button(self.control_frame, text="Demo (10 Diamond)", command=self.set_demo_mode, width=15, bg="#e67e22", fg="white")
        self.btn_demo.grid(row=0, column=1, padx=5)

        # Update tampilan awal
        self.update_ui()

    def get_symbol_color(self, card_name):
        # Helper untuk menentukan simbol dan warna
        if "Diamond" in card_name: return "♦", "red"
        if "Heart" in card_name: return "♥", "red"
        if "Spade" in card_name: return "♠", "black"
        if "Club" in card_name: return "♣", "black"
        return "?", "black"

    def update_ui(self):
        # 1. Update Teks Kartu
        if self.game.kartu_sekarang:
            nama_kartu = self.game.kartu_sekarang
            simbol, warna = self.get_symbol_color(nama_kartu)
            angka = nama_kartu.split(" ")[0]

            self.lbl_card_symbol.config(text=simbol, fg=warna)
            self.lbl_card_text_top.config(text=angka, fg=warna)
            self.lbl_card_text_bottom.config(text=angka, fg=warna)
        
        # 2. Update Skor & Pesan
        self.lbl_score.config(text=f"Score: {self.game.skor}")
        self.lbl_status.config(text=self.game.pesan_status)

        # Ubah warna pesan jika Benar/Salah
        if "BENAR" in self.game.pesan_status:
            self.lbl_status.config(fg="#2ecc71") # Hijau
        elif "SALAH" in self.game.pesan_status:
            self.lbl_status.config(fg="#e74c3c") # Merah
        else:
            self.lbl_status.config(fg="white")

    def proses_tebakan(self, tebakan):
        # Panggil logika otak
        self.game.cek_tebakan(tebakan)
        # Update wajah aplikasi
        self.update_ui()

        if self.game.game_selesai:
            messagebox.showinfo("Game Over", f"Permainan Selesai!\nSkor Akhir: {self.game.skor}")
            self.reset_game()

    def reset_game(self):
        self.game.mulai_game_baru()
        self.game.pesan_status = "Game Baru Dimulai!"
        self.update_ui()

    def set_demo_mode(self):
        # Fitur Khusus "Pak Pres" untuk Demo
        self.reset_game()
        self.game.kartu_sekarang = "10 Diamond" # Paksa kartu
        self.game.pesan_status = "DEMO MODE: Start 10 Diamond"
        self.update_ui()

# ==============================================================================
# MAIN LOOP
# ==============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = GameGUI(root)
    root.mainloop()