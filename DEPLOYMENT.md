# AutoDocumentScanner v1.0.0

## Paket distribusi Windows

Gunakan folder hasil release atau file ZIP yang dibuat oleh `release_windows.ps1`.
Paket final hanya berisi runtime aplikasi, dependency hasil PyInstaller, folder `input` dan `output` yang dikosongkan, serta metadata release.

Jangan menyalin seluruh repository ke PC induk. Yang dipindahkan cukup paket release final.

## Menjalankan aplikasi

1. Ekstrak paket ZIP ke folder lokal PC, misalnya `D:\AutoDocumentScanner`.
2. Jalankan `AutoDocumentScanner.exe`.
3. Pilih foto atau folder melalui UI.
4. Pilih mode output Warna atau Grayscale.
5. Tekan `Proses Otomatis`.
6. Hasil tersimpan di folder output yang dipilih. Jika tidak memilih folder lain, aplikasi memakai folder `output` di samping EXE.

Aplikasi tidak memerlukan Python pada PC tujuan karena runtime dibundel oleh PyInstaller.

## Struktur paket

```text
AutoDocumentScanner-v1.0.0-windows-x64\
|-- AutoDocumentScanner.exe
|-- _internal\
|-- input\
|-- output\
|-- VERSION.txt
|-- DEPLOYMENT.md
`-- SHA256SUMS.txt
```

Folder `input` dan `output` pada paket release harus kosong. Data pengujian lokal, hasil UAT, source code, repository Git, dan folder `tests` tidak ikut ke paket final.

## UAT PC induk

Setelah dipindahkan ke PC induk, lakukan pengujian singkat sebelum dipakai operasional:

- aplikasi dapat dibuka tanpa Python;
- UI tampil normal;
- satu foto KTP yang representatif dapat diproses;
- popup loading tampil selama proses;
- hasil perspective/crop sesuai baseline;
- mode Warna dan Grayscale dapat digunakan;
- file hasil dapat ditulis ke folder output;
- aplikasi dapat ditutup dan dibuka kembali tanpa error.

## Integritas paket

`SHA256SUMS.txt` berisi checksum SHA-256 file di dalam folder release. File ZIP final juga memiliki file checksum terpisah dengan akhiran `.sha256.txt`.

Jika paket dipindahkan melalui flashdisk atau jaringan internal dan ingin memastikan file tidak berubah, bandingkan checksum setelah pemindahan.
