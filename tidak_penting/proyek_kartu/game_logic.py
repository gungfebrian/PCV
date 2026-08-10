import random

class HigherLowerGame:
    def __init__(self):

        # Definisikan data kartu
        self.ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        self.suits = ["Spade", "Heart", "Diamond", "Club"]
        self.values = {
            "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, 
            "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13
        }
        
        # Inisiasi status game
        self.score = 0
        self.is_game_over = False
        self.current_card = None # Kartu acuan (dari random deck)
        self.next_card = None    # Kartu yang akan dibandingkan (dari CV)
        self.message = "Tekan 'R' untuk Mulai/Reset Game!"
        self.reveal_current_card = False # <--- FLAG INI MENGONTROL PENUTUPAN
        
        self.deck = self.create_deck() 

    def create_deck(self):
        """Membuat satu set kartu lengkap (52 kartu)."""
        deck = []
        for suit in self.suits:
            for rank in self.ranks:
                deck.append(f"{rank} {suit}")
        return deck

    def shuffle_deck(self):
        """Mengacak deck saat game dimulai atau direset."""
        self.deck = self.create_deck() # Buat deck baru
        random.shuffle(self.deck)     # Acak

    def get_value(self, card_string):
        """Mengambil nilai numerik dari string kartu (misal 'K Diamond' -> 13)."""
        if not card_string:
            return 0
        parts = card_string.split(" ")
        rank = parts[0]
        return self.values.get(rank, 0)

    # --- RESET DAN START GAME (MENGAMBIL KARTU RANDOM PERTAMA) ---
    def start_game(self):
        """Mengambil kartu acak pertama dan memulai game."""
        self.score = 0
        self.is_game_over = False
        self.shuffle_deck() 
        self.next_card = None 
        self.reveal_current_card = False # <--- SEMBUNYIKAN KARTU ACUAN BARU
        
        if self.deck:
            self.current_card = self.deck.pop() 
            self.message = "Kartu Acuan BARU diambil. Letakkan Kartu TEBAKAN Anda!"
        else:
            self.current_card = None
            self.message = "Deck kosong. Tekan 'R' untuk Reset."

    # --- FUNGSI BARU UNTUK TRANSISI KE RONDE BERIKUTNYA ---
    def start_new_round_card(self):
        """Mengambil kartu acak baru dari deck untuk memulai ronde berikutnya."""
        if self.is_game_over:
            self.message = "Game Selesai! Tekan 'R' untuk Mulai Ulang."
            return

        # Setelah pemain menekan tombol 'N' (Next), barulah kita pindah ke kartu acak berikutnya.
        if self.deck:
            self.current_card = self.deck.pop() 
            self.next_card = None # Reset next card
            self.reveal_current_card = False # <--- SEMBUNYIKAN KARTU ACUAN BARU
            self.message = "Kartu Acuan BARU diambil. Sekarang letakkan Kartu TEBAKAN Anda!"
        else:
            self.is_game_over = True
            self.message = f"Deck Internal Habis! Final Score: {self.score}. Tekan 'R' untuk Mulai Ulang."

    # --- JEMBATAN DARI COMPUTER VISION (CV) ---
    def set_card_from_cv(self, detected_card_name):
        """
        CV HANYA akan mengisi next_card. Diblokir jika game dalam mode Review.
        """
        if self.is_game_over:
            return
        
        # BLOKIR UPDATE: Jika pesan menunjukkan pemain harus menekan 'N', jangan update next_card
        if "Tekan 'N'" in self.message:
            return
        
        if self.current_card is None:
             self.message = "Tekan 'R' untuk memulai game (Ambil Kartu Acak Pertama)."
             return

        # Pastikan kartu yang terdeteksi bukan kartu yang sama
        if detected_card_name != self.current_card and detected_card_name != "tidak diketahui":
            
            # Isi next_card
            if self.next_card != detected_card_name:
                self.next_card = detected_card_name
                # Pesan ini hanya mengonfirmasi deteksi, bukan nama kartu.
                self.message = "Kartu TEBAKAN Terdeteksi. SILAHKAN TEBAK (H/L)"
        
        elif detected_card_name == self.current_card:
             self.message = f"Kartu Masih Sama: {self.current_card}. Ganti kartu untuk tebakan."


    # --- LOGIKA KEPUTUSAN (DIPICU TOMBOL H/L) ---
    def check_guess(self, guess):
        
        if self.is_game_over:
            return False

        # Cek ketersediaan kartu
        if self.current_card is None:
            self.message = "Tunggu Kartu Acuan terdeteksi! Tekan 'R'."
            return False
        
        if self.next_card is None:
            self.message = "Tunggu Kartu TEBAKAN terdeteksi (Ganti Kartu Anda)!"
            return False

        # Ambil nilai
        current_val = self.get_value(self.current_card)
        next_val = self.get_value(self.next_card)
        
        correct = False
        
        # Logika Tie (Nilai sama)
        if next_val == current_val:
            correct = True # Dianggap benar agar game lanjut
        # Logika Higher/Lower
        elif guess == "higher" and next_val > current_val:
            correct = True
        elif guess == "lower" and next_val < current_val:
            correct = True
        
        # Pembaruan Status dan Pesan HASIL (Reveal Kartu Acuan)
        self.reveal_current_card = True # <--- TUNJUKKAN KARTU ACUAN SETELAH DITEBAK!

        if correct:
            self.score += 1
            
            # 🏆 KONDISI MENANG: SKOR MENCAPAI 4
            if self.score >= 4:
                self.is_game_over = True
                self.message = f"Yeyy Menang! SKOR {self.score}. Kartu adalah {self.next_card} !!! 🎉"
            else:
                # Pesan Correct Final yang ditunggu
                self.message = f"TEBAKAN BENAR! Kartu adalah: {self.next_card}. +1 Score. Tekan 'N' untuk Lanjut."
        else:
            self.score = max(0, self.score - 1)
            # Pesan Wrong Final yang ditunggu
            self.message = f"TEBAKAN SALAH! Kartu adalah: {self.next_card}. Skor -1. Tekan 'N' untuk Lanjut."

        # PENTING: TIDAK ADA KODE TRANSISI DI SINI! (Mempertahankan hasil untuk review)
        
        return correct