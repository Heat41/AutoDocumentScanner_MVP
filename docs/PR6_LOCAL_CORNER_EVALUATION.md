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


## Analisis tambahan setelah evaluasi awal

Laporan menampilkan kesalahan per sudut **TL/TR/BR/BL** untuk lima
kandidat terdekat. Setiap sudut memuat jarak ke acuan manual, serta
selisih (dx,dy) dalam piksel foto asli. dx positif berarti kandidat
di sebelah kanan titik acuan, dy positif berarti di bawahnya.
Ini membantu membedakan satu sudut meleset, satu sisi salah, atau
seluruh kotak bergeser. Urutan sudut tetap dipetakan ke label manual
walaupun titik pertama kandidat berbeda atau foto diputar.

Laporan juga menampilkan jumlah usulan Side Recovery, Paired Alignment
dan Independent Sides, jumlah kandidat asal yang diuji, serta jumlah
hipotesis yang disembunyikan oleh batas tampilan. Jika sebuah keluarga
tidak terlihat, periksa jumlah pembangkitan ini sebelum menyimpulkan
bahwa algoritma gagal. Nilai terbaik tetap ORACLE pasca-anotasi,
tidak dapat digunakan sebagai ambang otomatis.


## Evaluasi langsung kandidat Outer Region

Independent Sides sebelumnya hanya mencoba keluaran Side Recovery.
Sekarang setiap Outer Region valid diperiksa langsung pada dua pasangan
sisi berhadapan (atas/bawah dan kiri/kanan), lalu keluaran Side Recovery
tetap diperiksa. Perubahan dilakukan **setelah** seluruh PR6 Review
Gate dan manual fallback. Tidak ada perubahan pada crop otomatis.

Penyaringan duplikat khusus hasil Independent Sides dikurangi dari
1,8% menjadi 0,3% diagonal gambar agar pergeseran kecil tetap
tersedia untuk inspeksi. Ambang bukti tepi dan geometri TIDAK
dilonggarkan. Hasil bisa tetap nol.

Pilih kandidat Outer Region dalam Visual Candidate Debugging.
Lihat bagian **INDEPENDENT SIDES LANGSUNG DARI OUTER REGION**
untuk bukti awal, puncak, ambang dan alasan penolakan pada kedua
pasangan sisi. Indeks v2_N adalah lokal per pemeriksaan gambar;
jangan menyamakan v2_2 antar foto.


## Sisi sudah kuat tidak perlu dipaksa bergeser

Independent Sides lama mensyaratkan KEDUA sisi berhadapan mengalami
pergeseran dengan kenaikan skor tepi yang besar. Jika satu tepi kartu
sudah kuat pada kandidat Outer Region awal, aturan ini berpotensi
menolak perbaikan yang hanya diperlukan di sisi lainnya.

Percobaan tambahan **one-side-held-one-shifted** memperbolehkan
satu sisi tetap di offset nol **hanya ketika skor tepi awal >=11**.
Sisi lawan tetap WAJIB memiliki pergeseran dengan bukti tepi lebih
kuat, dan kedua sisi diperiksa ulang sesudah penggabungan geometri.
Kandidat tanpa perpindahan pada kedua sisi dibuang. Hasil tetap
visual-review only, tidak mengubah scoring, pemilihan PR6, crop,
manual fallback, atau foto sumber.

Pada Visual Candidate Debugging, sisi yang dipertahankan diberi
label DITAHAN; diagnostik juga menginformasikan apabila bukti awal
belum cukup untuk mempertahankan sisi tersebut. Hasil tidak dijamin
muncul pada foto nyata, apalagi benar-benar mengikuti batas fisik
KTP: syarat warna/kontras saja belum membuktikan identitas kartu.
