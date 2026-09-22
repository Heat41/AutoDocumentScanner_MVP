# KTP Field Detector

Folder ini adalah lokasi model object detection lokal untuk Tracking Data KTP.

## Runtime final

File model final:

```
models/ktp_field_detector/ktp_fields.onnx
```

Aplikasi memuat model dengan OpenCV DNN pada CPU. Tidak ada download model saat runtime dan tidak ada dependency cloud.

Jika file `ktp_fields.onnx` belum tersedia, Tracking otomatis memakai template/anchor fallback supaya aplikasi tetap dapat dijalankan selama fase pengembangan dataset.

## Class order

Urutan class HARUS sama dengan `classes.txt`:

1. provinsi
2. kabupaten_kota
3. nik
4. nama
5. ttl
6. jenis_kelamin
7. golongan_darah
8. alamat
9. rt_rw
10. kelurahan_desa
11. kecamatan
12. agama
13. status_perkawinan
14. pekerjaan
15. kewarganegaraan
16. berlaku_hingga
17. foto

Detector dilatih untuk mendeteksi area VALUE, bukan tulisan label. Contoh: class `nama` membungkus `DJONG FUK HIE`, bukan kata `Nama`.

## Expected ONNX output

Runtime saat ini menerima output YOLO-style per detection:

```
[cx, cy, width, height, class_score_0, ..., class_score_16]
```

Output dapat berbentuk `[1, N, 21]` atau `[1, 21, N]`. Koordinat mengacu ke input model 640x640 dan akan dipetakan kembali ke corrected KTP.

Model training/export belum disertakan sebelum dataset KTP selesai dianotasi.
