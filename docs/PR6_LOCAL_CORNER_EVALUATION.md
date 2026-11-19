# PR6 — Evaluasi lokal empat sudut

## Tujuan
Setelah Review PR6, gunakan sudut fisik yang ditandai sendiri pada foto
ORIGINAL sebagai acuan untuk menghitung kesalahan kandidat. Ini adalah
evaluasi pasca-anotasi, bukan izin untuk auto-crop dan bukan bukti
generalisasi ke seluruh foto KTP.

## Langkah Windows
1. Aktifkan AUTODOCSCANNER_PR6_RUNTIME_REVIEW=1 dan jalankan app.py --ui.
2. Pilih foto, klik Review PR6 dan tutup jendela kandidat.
3. Klik Koreksi Manual dan geser KEEMPAT sudut ke batas luar fisik KTP.
4. Klik **Ukur Kandidat PR6**, bukan Terapkan Koreksi.
5. Lihat kesalahan rata-rata dalam piksel dan sebagai % diagonal FOTO asli,
   serta nilai terbaik, median dan p90 per keluarga deteksi.
6. Tutup hasilnya; pengukuran dan Batal tidak menyimpan foto, koordinat
   referensi, JSON atau crop.

Nilai terbaik per keluarga adalah **oracle setelah melihat anotasi**,
bukan pemilihan otomatis yang dilakukan detector. Foto yang sudah digunakan
untuk pengembangan hanya mengukur kegagalan yang diketahui. Pengujian
terhadap unseen holdout baru dengan label independen tetap diperlukan.

Koreksi Manual biasa, Proses Otomatis, PDF/JPG, SnipSheet dan branch main
tetap tidak diubah oleh fitur evaluasi. Tidak ada file yang diunggah.
