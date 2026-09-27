# Webcam Hand Gesture Filter: Love, Bunga Background, Persegi Twist, 3D Kristal, & Ketapel Blur

Filter webcam real-time interaktif berbasis **OpenCV** dan **MediaPipe Tasks (HandLandmarker & Selfie Segmenter)**.

## Fitur & Efek
1. **Tangan Kanan Nunjuk ke Atas** 👉❤️
   - Menghasilkan partikel hati (`assets/heart.png`) yang melayang ke atas dengan efek denyut (*pulse*), goyangan halus (*wobble*), kilauan (*sparkle*), dan transparansi (*fade-out*).
   - **Membuka (Unlock)** izin untuk efek bunga.
2. **Jempol Nunjuk ke Diri Sendiri** 👍🌸
   - **Bunga FULLSCREEN di BELAKANG SUBJEK**: Bunga menghujani seluruh layar (*fullscreen*) dan melayang anggun **di belakang tubuh pengguna** menggunakan AI *Selfie Segmentation*!
   - Tubuh pengguna tetap terlihat jelas di lapisan depan (*foreground*), sementara sakura, kelopak bunga, dan kilauan magis melayang di lapisan latar belakang (*background*).
   - **Syarat**: Hanya aktif **setelah** Anda melakukan Gesture 1 (Tangan Kanan Nunjuk ke Atas).
3. **Sentuh Telunjuk Kiri & Kanan 👉👈 lalu Melebar** 📐✨
   - Shape persegi panjang **mulai muncul saat ujung jari telunjuk kanan dan kiri bersentuhan**.
   - Setelah bersentuhan, saat kedua tangan melebar (*stretch*), terbentuk **shape persegi panjang neon bersih** yang sudut-sudutnya mengunci pada telunjuk dan jempol.
   - **Bisa di-Twist / Diputar 360°**: Mengikuti putaran tangan Anda (horizontal, diagonal, atau vertikal).
   - Dilengkapi *glow neon border*, *corner brackets*, lapisan kaca semi-transparan, dan taburan partikel kilauan.
4. **Telapak Bersentuhan di Bawah & Jari Merenggang Mekar** 💎✨
   - Rapatkan kedua pergelangan/pangkal telapak tangan di bagian bawah dan buka/lebarkan jari-jari ke atas (seperti membentuk mangkuk / teratai penopang kristal).
   - Memunculkan **Objek 3D Kristal Prisma Multi-Facet per Jari**:
     - Memiliki **9 sisi facet kristal** untuk setiap jari, menopang seluruh 10 jari.
     - Rusuk neon glow menyala di setiap jari dan cincin simpul bercahaya di setiap ujung jari.
     - Mengikuti deformasi dan pergerakan tangan secara real-time.
5. **Efek Ketapel ke Kamera & Usap Kamera Jadi Bersih** 🏹🌫️🖐️
   - **Pose Ketapel**: Satu tangan membentuk cabang ketapel (pose 'V' murni) dan tangan satunya mencubit (*pinch*) karet ketapel lalu menariknya ke belakang (*pull & stretch*).
   - Muncul tali karet neon elastis bergetar dan peluru energi bercahaya di kantung ketapel dengan indikator daya tarikan (0% - 100%).
   - **Lepaskan Cubitan**: Peluru energi melesat kencang menuju lensa kamera (3D zoom, radial speed lines, trail particles).
   - **Tabrakan (Impact)**: Guncangan layar (*screen shake*), kilatan cahaya (*white flash*), dan seluruh tampilan kamera berubah menjadi **kaca berembun / frosted glass blur**!
   - **Usap Kamera Jadi Bersih**: Cukup usapkan telapak tangan Anda di depan kamera seperti mengusap kaca jendela berembun. Sapuan tangan Anda akan menghapus embun secara real-time dengan percikan air berkilau hingga kamera jernih kembali 100%!
6. **Dua Tangan Menarik dari Atas ke Bawah -> Filter Black & White (Push dari Atas)** 🎬⬇️
   - Letakkan kedua tangan di bagian atas lalu gerakkan menarik ke bawah secara bersamaan (*two-hand downward pull*).
   - Memicu **animasi transisi tirai/laser push dari atas ke bawah**:
     - Garis laser pembatas neon bercahaya menyapu dari $y=0$ turun ke bawah dengan taburan percikan kilau (*laser sparks*).
     - Mengubah kamera menjadi **filter Black & White (B&W) monokrom sinematik** dengan kontras tinggi ala film noir.
   - **Toggle Reversibel**: Menarik kedua tangan ke bawah lagi akan mendorong (*push*) kembali filter warna asli dari atas ke bawah!

---

## Cara Menjalankan

Buka terminal / PowerShell di folder ini, lalu jalankan:

```bash
python hand_effects.py
```

### Tombol Kontrol:
- **`ESC`** atau **`Q`** : Keluar dari aplikasi.
- **`B`** : Toggle manual filter Black & White (B&W).
- **`R`** : Reset semua gesture, status unlock, embun kamera, dan filter B&W.
- **`L`** : Tampilkan/sembunyikan titik landmark tangan (mode debug).
- **`C`** : Bersihkan partikel yang sedang aktif di layar.

---

## Struktur File
- `hand_effects.py` : Script filter utama kamera.
- `assets/`
  - `heart.png` : Gambar partikel love/hati.
  - `flower_sakura.png` : Sprite bunga sakura.
  - `flower_bloom.png` : Sprite bunga mekar.
  - `flower_petal.png` : Sprite kelopak bunga berguguran.
- `test_effects.py` : Automated test suite untuk seluruh 6 fitur gesture dan rendering efek.
- `hand_landmarker.task` : Model AI deteksi tangan dari Google MediaPipe.
- `selfie_multiclass.tflite` : Model AI segmentasi tubuh untuk efek latar belakang.
