from pathlib import Path


class BatchScanRunner:
    """Menjalankan banyak file tanpa membiarkan satu kegagalan menghentikan batch.

    Runner ini tidak mengubah pipeline scanner. Ia hanya mengisolasi eksekusi
    per-file dan membuat snapshot metadata agar hasil sukses/gagal dari setiap
    file dapat dilaporkan secara konsisten oleh CLI maupun UI.
    """

    def __init__(self, scanner):
        self.scanner = scanner

    @staticmethod
    def _snapshot(scanner):
        return {
            "metadata": dict(
                getattr(scanner, "last_detection", {}) or {}
            ),
            "quality": dict(
                getattr(scanner, "last_quality", {}) or {}
            ),
            "validation": dict(
                getattr(scanner, "last_validation", {}) or {}
            ),
        }

    def run_one(
        self,
        input_path,
        output_path,
        output_mode="color",
        mode="ktp",
    ):
        input_path = Path(input_path)
        output_path = Path(output_path)

        try:
            _, corners = self.scanner.scan(
                input_path,
                output_path,
                mode=mode,
                output_mode=output_mode,
            )

            snapshot = self._snapshot(self.scanner)

            return {
                "success": True,
                "input_path": input_path,
                "output_path": output_path,
                "corners": (
                    corners.copy()
                    if hasattr(corners, "copy")
                    else corners
                ),
                "error": "",
                **snapshot,
            }

        except Exception as exc:
            snapshot = self._snapshot(self.scanner)
            corners = getattr(
                self.scanner,
                "last_corners",
                None,
            )

            return {
                "success": False,
                "input_path": input_path,
                # Penting: path lama tidak dianggap sebagai hasil run terbaru.
                "output_path": None,
                "attempted_output_path": output_path,
                "corners": (
                    corners.copy()
                    if hasattr(corners, "copy")
                    else corners
                ),
                "error": str(exc),
                **snapshot,
            }

    def run_many(
        self,
        input_paths,
        output_dir,
        output_mode="color",
        mode="ktp",
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        results = []

        for input_path in input_paths:
            input_path = Path(input_path)
            output_path = (
                output_dir
                / f"{input_path.stem}_scanned.jpg"
            )

            results.append(
                self.run_one(
                    input_path,
                    output_path,
                    output_mode=output_mode,
                    mode=mode,
                )
            )

        success_count = sum(
            1
            for result in results
            if result["success"]
        )
        failed_count = len(results) - success_count

        return {
            "results": results,
            "success_count": success_count,
            "failed_count": failed_count,
            "total_count": len(results),
        }
