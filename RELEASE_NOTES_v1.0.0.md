# AutoDocumentScanner v1.0.0

Release internal pertama AutoDocumentScanner untuk Windows x64.

## Fitur utama

- Automatic perspective correction untuk KTP.
- Deteksi empat sudut fisik dengan fallback robustness.
- Auto orientation konservatif.
- Output warna default dan opsi grayscale.
- Quality gate dan final validation.
- Safe atomic output agar file lama tidak rusak saat proses gagal.
- Batch processing dengan isolasi kegagalan per file.
- UI desktop final dengan preview original/corner dan hasil scanner.
- Popup loading/progress selama proses.
- Packaging Windows CPU-only tanpa membutuhkan Python di PC tujuan.
- Installer Windows dan ZIP portable.

## Paket

- `AutoDocumentScanner-v1.0.0-windows-x64.zip` — versi portable.
- `AutoDocumentScanner-v1.0.0-Setup.exe` — installer Windows.
- File `.sha256.txt` tersedia untuk verifikasi integritas.

## Catatan deployment

Paket release tidak membawa foto contoh, hasil UAT lokal, source code, folder `tests`, maupun isi runtime `input/output` dari mesin build.

Setelah instalasi atau ekstrak ZIP, lakukan UAT singkat di PC tujuan sebelum digunakan operasional.
