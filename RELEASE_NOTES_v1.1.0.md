# AutoDocumentScanner v1.1.0

Release ini memfokuskan penyempurnaan pengalaman desktop dan kesiapan rilis publik untuk alur Auto KTP dan Dokumen Manual.

## Perubahan utama

- UI desktop diperbarui dengan tema terang/gelap, sidebar yang dapat diperkecil, header/status yang konsisten, dan layout responsif.
- Auto KTP kini mendukung drag & drop yang aman, popup mode Warna/Grayscale/B&W, preview yang menyesuaikan ukuran window, serta alur tombol yang lebih jelas.
- Output Auto KTP dipisahkan menjadi `Simpan` dan `Simpan PDF`.
- PDF KTP menggunakan lembar A4 dengan maksimal 6 kartu per halaman tanpa mengubah rasio kartu.
- Dokumen Manual dipoles dengan area upload baru, navigasi halaman, empat titik sudut, rotasi, mode Warna/Grayscale/B&W, `Terapkan Ulang`, serta ekspor gambar/PDF.
- Empty state, status proses, tombol disabled/enabled, preview hasil, dan posisi label sudut diperbaiki.
- Tracking KTP tetap tersedia di codebase untuk pengembangan lanjutan, tetapi pada release publik ditampilkan sebagai `Coming Soon`.
- Native Windows title bar tetap dikelola sistem untuk menjaga tombol minimize/maximize/close stabil.
- Responsive breakpoint diselaraskan agar mode narrow/compact/wide konsisten.
- Full regression baseline sebelum release: 299 test lulus.

## Paket

- `AutoDocumentScanner-v1.1.0-windows-x64.zip` — versi portable.
- `AutoDocumentScanner-v1.1.0-Setup.exe` — installer Windows.
- File `.sha256.txt` tersedia untuk verifikasi integritas paket.

## Catatan deployment

Paket release tidak menyertakan foto contoh, hasil UAT lokal, source code, folder `tests`, maupun isi runtime `input/output` dari mesin build.

Setelah instalasi atau ekstrak ZIP, lakukan UAT singkat pada PC tujuan: buka aplikasi, resize/maximize/restore, proses satu KTP, uji Dokumen Manual, simpan gambar/PDF, dan pastikan Tracking KTP tetap berstatus Coming Soon.
