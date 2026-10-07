# AutoDocumentScanner v1.1.1

Release patch ini memfokuskan peningkatan ketahanan Auto KTP pada foto random, menjaga bentuk hasil tetap sesuai rasio fisik KTP, dan menyediakan fallback koreksi manual tanpa mengubah alur otomatis utama.

## Perubahan utama

- Auto perspective KTP dibuat lebih adaptif untuk berbagai foto random, termasuk background bermotif, frame luar yang kuat, dan kondisi kartu yang tidak seragam.
- Kandidat outer-frame yang masih memuat bentuk KTP di dalamnya kini diturunkan skornya agar kartu fisik lebih diprioritaskan.
- Robustness engine tidak lagi langsung mengunci baseline yang mencurigakan hanya karena confidence tinggi.
- Hasil KTP dinormalisasi ke rasio fisik ID-1 85.60 × 53.98 mm.
- Ditambahkan safe margin dinamis pada warp otomatis untuk mengurangi risiko bagian tepi KTP terpotong.
- Preview Auto KTP dibuat bersih tanpa garis/marker corner; koordinat tetap dipakai internal.
- Ditambahkan tombol Koreksi Manual sebagai fallback setelah proses otomatis.
- Koreksi Manual menggunakan empat titik hasil auto sebagai titik awal, dapat digeser pengguna, dapat di-reset ke posisi otomatis, dan hasilnya tetap melalui pipeline validasi/output KTP.
- Bug canvas putih saat titik koreksi manual digeser sudah diperbaiki.
- Pytest discovery dibatasi hanya ke test project agar folder release/build tidak ikut dikoleksi.

## Validasi

- Uji manual dilakukan pada beberapa foto KTP berbeda.
- Full regression project: 239 passed, 0 failed.

## Paket

- `AutoDocumentScanner-v1.1.1-windows-x64.zip` — versi portable.
- `AutoDocumentScanner-v1.1.1-Setup.exe` — installer Windows.
- File `.sha256.txt` disiapkan untuk verifikasi integritas paket.

## Catatan deployment

Paket release tidak menyertakan foto contoh, hasil UAT lokal, source code, folder `tests`, maupun isi runtime `input/output` dari mesin build.

Setelah instalasi atau ekstrak ZIP, lakukan UAT singkat: buka aplikasi, proses beberapa KTP dengan kondisi foto berbeda, pastikan hasil auto tidak terpotong, uji Koreksi Manual, simpan gambar/PDF, dan pastikan Tracking KTP tetap berstatus Coming Soon.
