# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files


# OpenCV's Haar cascade XML files are required by scanner.py for conservative
# KTP orientation detection. Bundle cv2 package data explicitly so the frozen
# app does not depend on a system OpenCV installation.
cv2_datas = collect_data_files("cv2")
brand_datas = [
    ("assets/logo.png", "assets"),
    ("assets/logo.ico", "assets"),
]


a = Analysis(
    ["desktop_launcher.py"],
    pathex=[],
    binaries=[],
    datas=cv2_datas + brand_datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AutoDocumentScanner",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon="assets/logo.ico",
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AutoDocumentScanner",
)
