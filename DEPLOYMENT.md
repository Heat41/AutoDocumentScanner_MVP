# AutoDocumentScanner v1.1.0

## Paket distribusi Windows

Ada dua bentuk distribusi final:

1. **ZIP portable** untuk dipindahkan dan dijalankan tanpa proses instalasi.
2. **Installer Windows** untuk pemasangan normal dengan Start Menu, uninstaller, dan shortcut Desktop opsional.

Keduanya dibangun dari paket release yang sama dan tidak menyertakan source code, folder `tests`, foto contoh, ataupun hasil UAT.

Jangan menyalin seluruh repository ke PC induk. Yang dipindahkan cukup ZIP portable atau installer final.

## Opsi 1 - ZIP portable

Gunakan ZIP yang dibuat oleh `release_windows.ps1`.

1. Ekstrak ZIP ke folder lokal PC, misalnya `D:\AutoDocumentScanner`.
2. Jalankan `AutoDocumentScanner.exe`.
3. Pilih foto atau folder melalui UI.
4. Pilih mode proses Warna, Grayscale, atau B&W saat diminta.
5. Tekan `Proses Otomatis`.
6. Gunakan `Simpan` untuk gambar atau `Simpan PDF` untuk PDF KTP.
7. Hasil tersimpan di lokasi yang dipilih melalui dialog simpan.

Aplikasi tidak memerlukan Python pada PC tujuan karena runtime dibundel oleh PyInstaller.

## Opsi 2 - Installer Windows

Installer dibuat dengan Inno Setup 6 melalui `build_installer.ps1`.

Installer memakai instalasi **per-user** ke `%LOCALAPPDATA%\Programs\AutoDocumentScanner`, sehingga tidak memerlukan hak Administrator dan aplikasi tetap mempunyai akses tulis ke folder runtime `input` dan `output`.

Installer menyediakan:

- shortcut Start Menu;
- shortcut Desktop opsional;
- entry uninstall Windows;
- opsi menjalankan aplikasi setelah instalasi;
- update di lokasi instalasi yang sama untuk versi berikutnya.

Folder `input` dan `output` ditandai agar tidak otomatis dihapus saat uninstall, sehingga data runtime milik user tidak sengaja ikut terhapus.

## Membuat release final

### ZIP saja

```powershell
powershell -ExecutionPolicy Bypass -File .\release_windows.ps1
```

### Installer saja

Pastikan Inno Setup 6 sudah terpasang, lalu:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_installer.ps1
```

### ZIP + installer sekaligus

```powershell
powershell -ExecutionPolicy Bypass -File .\make_release.ps1
```

Semua artefak akhir dibuat di folder `release\`.

## Struktur ZIP portable

```text
AutoDocumentScanner-v1.1.0-windows-x64\
|-- AutoDocumentScanner.exe
|-- _internal\
|-- input\
|-- output\
|-- VERSION.txt
|-- DEPLOYMENT.md
|-- RELEASE_NOTES.md
`-- SHA256SUMS.txt
```

Folder `input` dan `output` pada paket release harus kosong. Data pengujian lokal, hasil UAT, source code, repository Git, dan folder `tests` tidak ikut ke paket final.

## File release yang dihasilkan

```text
release\
|-- AutoDocumentScanner-v1.1.0-windows-x64\
|-- AutoDocumentScanner-v1.1.0-windows-x64.zip
|-- AutoDocumentScanner-v1.1.0-windows-x64.zip.sha256.txt
|-- AutoDocumentScanner-v1.1.0-Setup.exe
`-- AutoDocumentScanner-v1.1.0-Setup.exe.sha256.txt
```

## UAT PC induk

Setelah dipindahkan ke PC induk, lakukan pengujian singkat sebelum dipakai operasional:

- aplikasi dapat dibuka tanpa Python;
- UI tampil normal dan responsif saat resize, maximize, dan restore;
- tombol `Proses Otomatis` tetap terlihat pada ukuran window yang didukung;
- satu foto KTP yang representatif dapat diproses;
- drag & drop KTP tidak menyebabkan aplikasi force close;
- popup loading tampil selama proses;
- hasil perspective/crop sesuai baseline;
- mode Warna, Grayscale, dan B&W dapat digunakan;
- `Simpan` dan `Simpan PDF` bekerja;
- PDF KTP mempertahankan proporsi kartu dan dapat menampung sampai 6 kartu per halaman A4;
- Dokumen Manual dapat menerima file, menggeser empat titik, rotate, menerapkan koreksi, dan ekspor gambar/PDF;
- state `Terapkan Ulang` muncul setelah koreksi Dokumen Manual;
- Tracking KTP pada release publik tetap tampil sebagai `Coming Soon`;
- aplikasi dapat ditutup dan dibuka kembali tanpa error.

Untuk installer, cek juga Start Menu, shortcut Desktop bila dipilih, serta menu uninstall Windows.

## Integritas paket

`SHA256SUMS.txt` berisi checksum SHA-256 file di dalam folder release portable. ZIP dan installer final masing-masing memiliki file checksum terpisah dengan akhiran `.sha256.txt`.

Jika paket dipindahkan melalui flashdisk atau jaringan internal dan ingin memastikan file tidak berubah, bandingkan checksum setelah pemindahan.
