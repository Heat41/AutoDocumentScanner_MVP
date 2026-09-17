import base64
import io
import sys
from pathlib import Path

from PIL import Image


ASSET_DIR_NAME = "assets"
LOGO_SOURCE_NAME = "logo_source.b64"
LOGO_PNG_NAME = "logo.png"
LOGO_ICO_NAME = "logo.ico"
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


def ensure_brand_assets(root=None):
    """Generate PNG/ICO branding assets from the tracked logo source."""
    root = Path(root) if root is not None else source_root()
    asset_dir = root / ASSET_DIR_NAME
    source_file = asset_dir / LOGO_SOURCE_NAME
    png_path = asset_dir / LOGO_PNG_NAME
    ico_path = asset_dir / LOGO_ICO_NAME

    if not source_file.exists():
        raise FileNotFoundError(f"Logo source tidak ditemukan: {source_file}")

    asset_dir.mkdir(parents=True, exist_ok=True)
    encoded = "".join(source_file.read_text(encoding="utf-8").split())
    raw_png = base64.b64decode(encoded, validate=True)

    if not png_path.exists() or png_path.read_bytes() != raw_png:
        png_path.write_bytes(raw_png)

    with Image.open(io.BytesIO(raw_png)) as image:
        image = image.convert("RGBA")
        image.save(
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
