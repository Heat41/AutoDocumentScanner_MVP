# AutoDocumentScanner v1.0.1

Bugfix release untuk penyempurnaan tampilan desktop pada berbagai ukuran dan resolusi layar Windows.

## Perubahan utama

- UI kini lebih responsif saat window di-resize, maximize, dan restore.
- Preview menyesuaikan ruang yang tersedia dan dapat berpindah ke layout vertikal pada window yang sempit.
- Sidebar menyesuaikan ukuran agar ruang preview tetap proporsional.
- Footer aksi dipertahankan agar tombol `Proses Otomatis` tidak terpotong pada layar yang lebih pendek atau display scaling tertentu.
- Ukuran awal window disesuaikan dengan ukuran layar agar tidak melewati area desktop yang terlihat.
- Branding/logo aplikasi tetap digunakan pada UI, EXE, shortcut, dan installer.
- Engine scanner, perspective correction, quality gate, final validation, dan safe output tidak diubah pada release ini.

## Paket

- `AutoDocumentScanner-v1.0.1-windows-x64.zip` — versi portable.
- `AutoDocumentScanner-v1.0.1-Setup.exe` — installer Windows.
- File `.sha256.txt` tersedia untuk verifikasi integritas paket.

## Catatan deployment

Paket release tidak menyertakan foto contoh, hasil UAT lokal, source code, folder `tests`, maupun isi runtime `input/output` dari mesin build.

Setelah instalasi atau ekstrak ZIP, lakukan UAT singkat pada PC tujuan, termasuk resize/maximize window dan proses satu dokumen representatif sebelum digunakan operasional.
