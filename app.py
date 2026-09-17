from pathlib import Path
import argparse

from batch_runner import BatchScanRunner
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


def _print_result(index, total, result):
    input_path = result["input_path"]

    print("--------------------------------------")
    print(
        f"[{index}/{total}] "
        f"{input_path.name}"
    )

    metadata = result.get("metadata") or {}
    quality = result.get("quality") or {}
    validation = result.get("validation") or {}

    if result["success"]:
        print("[OK] Perspective correction selesai.")
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

        if quality:
            print(
                f"Quality  : "
                f"{quality.get('status', '-')} "
                f"({quality.get('score', 0):.2f})"
            )

        if validation:
            print(
                f"Validasi : "
                f"{validation.get('status', '-')}"
            )

        print(
            f"Output   : {result['output_path']}"
        )
        return

    print(
        f"[FAILED] {result.get('error') or 'proses gagal'}"
    )

    warnings = list(
        validation.get("warnings") or []
    )
    if not warnings:
        warnings = list(
            quality.get("warnings") or []
        )

    if warnings:
        print(
            "Alasan   : "
            + "; ".join(
                str(item)
                for item in warnings[:3]
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
        from ui_safe import main as run_ui

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
    runner = BatchScanRunner(scanner)

    report = runner.run_many(
        images,
        OUTPUT_DIR,
        output_mode=output_mode,
        mode="ktp",
    )

    for index, result in enumerate(
        report["results"],
        start=1,
    ):
        _print_result(
            index,
            report["total_count"],
            result,
        )

    print("\n======================================")
    print(" SELESAI")
    print("======================================")
    print(
        f"Berhasil : {report['success_count']}"
    )
    print(
        f"Gagal    : {report['failed_count']}"
    )
    print(
        f"Output   : {OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
