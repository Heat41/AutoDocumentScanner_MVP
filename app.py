from pathlib import Path
import argparse

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


def get_input_images(folder):
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    return sorted(
        path
        for path in folder.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="AutoDocumentScanner MVP"
    )

    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Output grayscale. Default tetap warna.",
    )

    parser.add_argument(
        "--ui",
        action="store_true",
        help="Buka antarmuka desktop minimal.",
    )

    args = parser.parse_args()

    if args.ui:
        from ui import main as run_ui

        run_ui()
        return

    print("======================================")
    print(" AUTO DOCUMENT SCANNER")
    print("======================================")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = get_input_images(
        INPUT_DIR
    )

    if not images:
        print("\n[ERROR] Tidak ada gambar di folder input.")
        print(
            f"Folder: {INPUT_DIR.resolve()}"
        )
        return

    output_mode = (
        "grayscale"
        if args.grayscale
        else "color"
    )

    print(
        f"\nDitemukan {len(images)} gambar."
    )
    print(
        f"Mode output: {output_mode}\n"
    )

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
            scanner.scan(
                input_path,
                output_path,
                mode="ktp",
                output_mode=output_mode,
            )

            metadata = (
                scanner.last_detection
                or {}
            )

            print(
                "[OK] Perspective correction selesai."
            )
            print(
                f"Candidate: "
                f"{metadata.get('candidate_count', '-')}"
            )
            print(
                f"Source   : "
                f"{metadata.get('selected_source', '-')}"
            )
            print(
                f"Score    : "
                f"{metadata.get('score', 0):.3f}"
            )
            print(
                f"Output   : {output_path}"
            )

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
