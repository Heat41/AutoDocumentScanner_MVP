import base64
import hashlib
import io
import sys
from pathlib import Path

from PIL import Image


ASSET_DIR_NAME = "assets"
LOGO_SOURCE_DIR_NAME = "logo_source"
LOGO_PNG_NAME = "logo.png"
LOGO_ICO_NAME = "logo.ico"
LOGO_SHA256 = "147c9ceda6f5bba1b03a69f06455757e1066b9186905eef237280542f839333a"
LOGO_SIZE = (256, 256)
ICON_SIZES = (
    (16, 16),
    (24, 24),
    (32, 32),
    (48, 48),
    (64, 64),
    (128, 128),
    (256, 256),
)


def source_root():
    return Path(__file__).resolve().parent


def runtime_root():
    if getattr(sys, "frozen", False) and getattr(sys, "_MEIPASS", None):
        return Path(sys._MEIPASS)
    return source_root()


def _read_tracked_logo(asset_dir):
    source_dir = asset_dir / LOGO_SOURCE_DIR_NAME
    parts = sorted(source_dir.glob("part*.b64"))

    if not parts:
        raise FileNotFoundError(
            f"Logo source tidak ditemukan: {source_dir}"
        )

    encoded = "".join(
        "".join(part.read_text(encoding="utf-8").split())
        for part in parts
    )
    raw_png = base64.b64decode(encoded, validate=True)

    actual_hash = hashlib.sha256(raw_png).hexdigest()
    if actual_hash != LOGO_SHA256:
        raise ValueError(
            "Logo source gagal verifikasi SHA-256. "
            f"Expected {LOGO_SHA256}, got {actual_hash}."
        )

    return raw_png


def ensure_brand_assets(root=None):
    """Generate verified PNG/ICO branding assets from tracked source chunks."""
    root = Path(root) if root is not None else source_root()
    asset_dir = root / ASSET_DIR_NAME
    png_path = asset_dir / LOGO_PNG_NAME
    ico_path = asset_dir / LOGO_ICO_NAME

    asset_dir.mkdir(parents=True, exist_ok=True)
    raw_png = _read_tracked_logo(asset_dir)

    with Image.open(io.BytesIO(raw_png)) as image:
        if image.format != "PNG" or image.size != LOGO_SIZE:
            raise ValueError(
                "Logo source tidak sesuai format/ukuran resmi "
                f"({image.format}, {image.size})."
            )

        rgba = image.convert("RGBA")

    if not png_path.exists() or png_path.read_bytes() != raw_png:
        png_path.write_bytes(raw_png)

    rgba.save(
        ico_path,
        format="ICO",
        sizes=ICON_SIZES,
    )

    return {
        "png": png_path,
        "ico": ico_path,
    }


def brand_asset_paths():
    """Return usable asset paths in source mode or a PyInstaller bundle."""
    root = runtime_root()
    png_path = root / ASSET_DIR_NAME / LOGO_PNG_NAME
    ico_path = root / ASSET_DIR_NAME / LOGO_ICO_NAME

    if not getattr(sys, "frozen", False):
        return ensure_brand_assets(root)

    return {
        "png": png_path,
        "ico": ico_path,
    }


def main():
    paths = ensure_brand_assets()
    print(f"PNG : {paths['png']}")
    print(f"ICO : {paths['ico']}")


if __name__ == "__main__":
    main()
