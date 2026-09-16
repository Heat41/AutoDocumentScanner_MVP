from pathlib import Path
import cv2

from scanner import AutoDocumentScanner


INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_input_images():
    INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = []

    for file_path in INPUT_DIR.iterdir():
        if (
            file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        ):
            images.append(file_path)

    return sorted(images)


def main():
    print("======================================")
    print(" AUTO DOCUMENT SCANNER")
    print("======================================")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = get_input_images()

    if not images:
        print("\n[ERROR] Tidak ada gambar di folder input.")
        print(f"Folder: {INPUT_DIR.resolve()}")
        return

    print(f"\nDitemukan {len(images)} gambar.\n")

    scanner = AutoDocumentScanner()

    success_count = 0
    failed_count = 0

    for index, input_path in enumerate(
        images,
        start=1,
    ):
        print("--------------------------------------")
        print(
            f"[{index}/{len(images)}] "
            f"{input_path.name}"
        )

        output_path = (
            OUTPUT_DIR
            / f"{input_path.stem}_scanned.jpg"
        )

        try:
            result, corners = scanner.scan(
                input_path,
                output_path,
                mode="ktp",
            )

            print("[OK] Dokumen berhasil diproses.")
            print(f"Output: {output_path}")

            success_count += 1

        except Exception as exc:
            print(
                f"[FAILED] {exc}"
            )

            failed_count += 1

    print("\n======================================")
    print(" SELESAI")
    print("======================================")

    print(
        f"Berhasil : {success_count}"
    )

    print(
        f"Gagal    : {failed_count}"
    )

    print(
        f"Output   : {OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()