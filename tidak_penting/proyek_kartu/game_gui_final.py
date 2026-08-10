import tkinter as tk
from tkinter import messagebox
import random

# ==============================================================================
# 1. LOGIKA GAME (OTAK) - TETAP SAMA
# ==============================================================================
class GameTebakKartu:
    def __init__(self):
        self.daftar_angka = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        self.jenis_bunga  = ["Spade", "Heart", "Diamond", "Club"]
        
        self.kamus_nilai = {
            "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, 
            "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13
        }
        
        self.skor = 0
        self.game_selesai = False
        self.tumpukan_kartu = []
        self.kartu_sekarang = None
        self.kartu_selanjutnya = None
        self.pesan_status = "Game Siap!"
        
        self.mulai_game_baru()

    def buat_tumpukan_baru(self):
        return [f"{r} {s}" for s in self.jenis_bunga for r in self.daftar_angka]

    def mulai_game_baru(self):
        self.skor = 0
        self.game_selesai = False
        self.tumpukan_kartu = self.buat_tumpukan_baru()
        random.shuffle(self.tumpukan_kartu)
        
        if self.tumpukan_kartu:
            self.kartu_sekarang = self.tumpukan_kartu.pop()

    def terjemahkan_ke_angka(self, nama_kartu):
        if not nama_kartu: return 0
        return self.kamus_nilai.get(nama_kartu.split()[0], 0)

    def cek_tebakan(self, tebakan):
        if not self.tumpukan_kartu:
            self.game_selesai = True
            self.pesan_status = "Deck Habis!"
            return False

        self.kartu_selanjutnya = self.tumpukan_kartu.pop()
        val_lama = self.terjemahkan_ke_angka(self.kartu_sekarang)
        val_baru = self.terjemahkan_ke_angka(self.kartu_selanjutnya)
        
        benar = False
        if (tebakan == "tinggi" and val_baru > val_lama) or \
           (tebakan == "rendah" and val_baru < val_lama):
            benar = True
            
        if benar:
            self.skor += 1
            self.pesan_status = "BENAR! Lanjut..."
        else:
            self.skor = max(0, self.skor - 1)
            self.pesan_status = f"SALAH! Muncul: {self.kartu_selanjutnya}"
            
        self.kartu_sekarang = self.kartu_selanjutnya
        return benar

# ==============================================================================
# 2. UI TKINTER (MODERN & PROFESSIONAL)
# ==============================================================================
class GameApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Higher or Lower - Demo Leadership")
        self.root.geometry("400x650")
        self.root.configure(bg="#1e272e") # Dark theme profesional

        self.game = GameTebakKartu()
        
        # --- HACK DEMO (FIX 10 DIAMOND) ---
        self.game.kartu_sekarang = "10 Diamond"
        self.game.pesan_status = "DEMO START: 10 Diamond"

        # Header
        tk.Label(root, text="HIGHER OR LOWER", font=("Helvetica", 18, "bold"), 
                 fg="#ecf0f1", bg="#1e272e").pack(pady=20)
        
        self.lbl_skor = tk.Label(root, text=f"Skor: {self.game.skor}", font=("Helvetica", 14), 
                                 fg="#f1c40f", bg="#1e272e")
        self.lbl_skor.pack()

        # KARTU DISPLAY
        self.card_frame = tk.Frame(root, bg="white", width=220, height=320, relief="raised", bd=5)
        self.card_frame.pack(pady=20)
        self.card_frame.pack_propagate(False)

        # Elemen Kartu
        self.lbl_top = tk.Label(self.card_frame, text="10", font=("Arial", 24, "bold"), bg="white")
        self.lbl_top.place(x=10, y=10)
        
        self.lbl_center = tk.Label(self.card_frame, text="♦", font=("Arial", 100), bg="white")
        self.lbl_center.place(relx=0.5, rely=0.5, anchor="center")
        
        self.lbl_bottom = tk.Label(self.card_frame, text="10", font=("Arial", 24, "bold"), bg="white")
        self.lbl_bottom.place(x=150, y=270)

        # Status
        self.lbl_status = tk.Label(root, text=self.game.pesan_status, font=("Helvetica", 12), 
                                   fg="#ecf0f1", bg="#1e272e", wraplength=380)
        self.lbl_status.pack(pady=10)

        # Tombol
        btn_frame = tk.Frame(root, bg="#1e272e")
        btn_frame.pack(pady=20)

        tk.Button(btn_frame, text="▲ HIGHER", bg="#27ae60", fg="white", font=("Arial", 12, "bold"), width=12,
                  command=lambda: self.tebak("tinggi")).grid(row=0, column=0, padx=10)
        
        tk.Button(btn_frame, text="▼ LOWER", bg="#c0392b", fg="white", font=("Arial", 12, "bold"), width=12,
                  command=lambda: self.tebak("rendah")).grid(row=0, column=1, padx=10)

        # Tombol Reset Demo
        tk.Button(root, text="⟳ Reset Demo (10 Diamond)", bg="#e67e22", fg="white", 
                  command=self.reset_demo).pack(pady=10)

        self.update_ui()

    def get_visuals(self, nama_kartu):
        # Logika Warna & Simbol
        rank = nama_kartu.split()[0]
        if "Diamond" in nama_kartu: return "♦", "red", rank
        if "Heart" in nama_kartu:   return "♥", "red", rank
        if "Spade" in nama_kartu:   return "♠", "black", rank
        if "Club" in nama_kartu:    return "♣", "black", rank
        return "?", "black", rank

    def update_ui(self):
        simbol, warna, rank = self.get_visuals(self.game.kartu_sekarang)
        
        self.lbl_top.config(text=rank, fg=warna)
        self.lbl_center.config(text=simbol, fg=warna)
        self.lbl_bottom.config(text=rank, fg=warna)
        
        self.lbl_skor.config(text=f"Skor: {self.game.skor}")
        self.lbl_status.config(text=self.game.pesan_status)
        
        if "BENAR" in self.game.pesan_status: self.lbl_status.config(fg="#2ecc71")
        elif "SALAH" in self.game.pesan_status: self.lbl_status.config(fg="#e74c3c")
        else: self.lbl_status.config(fg="#ecf0f1")

    def tebak(self, pilihan):
        self.game.cek_tebakan(pilihan)
        self.update_ui()
        if self.game.game_selesai:
            messagebox.showinfo("Selesai", f"Skor Akhir: {self.game.skor}")
            self.reset_demo()

    def reset_demo(self):
        self.game.mulai_game_baru()
        self.game.kartu_sekarang = "10 Diamond" # FORCE RESET 10 DIAMOND
        self.game.pesan_status = "DEMO RESET: 10 Diamond"
        self.update_ui()

if __name__ == "__main__":
    root = tk.Tk()
    app = GameApp(root)
    root.mainloop()