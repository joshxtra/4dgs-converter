# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for 4DGS Converter."""

import sys

a = Analysis(
    ['app/converter/__main__.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('app/converter/icon.png', 'app/converter'),
        ('app/converter/icon.ico', 'app/converter'),
    ],
    hiddenimports=[
        'app.converter',
        'app.converter.main_window',
        'app.converter.worker',
        'app.converter.env_check',
        'app.pipeline',
        'app.pipeline.video_to_images',
        'app.pipeline.images_to_ply',
        'app.pipeline.ply_to_gsd',
        'app.pipeline.ply_to_gsd_v2',
        'app.pipeline.ply_to_raw',
        'app.pipeline.raw_to_gsd',
        'app.utils',
        'app.utils.ply_reader',
        'app.utils.morton',
        'lz4',
        'lz4.block',
        'numpy',
        'PIL',
        'sklearn',
        'sklearn.cluster',
        'sklearn.cluster._kmeans',
        'sklearn.utils._cython_blas',
        'sklearn.neighbors._partition_nodes',
        'app.utils.workers',
        'app.utils.gpu_kmeans',
        'psutil',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

if sys.platform == 'darwin':
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='4DGS-Converter',
        debug=False,
        strip=False,
        upx=True,
        console=False,
        icon='app/converter/icon.ico',
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        name='4DGS-Converter',
    )
    app = BUNDLE(
        coll,
        name='4DGS-Converter.app',
        icon='app/converter/icon.ico',
        bundle_identifier='com.dazaistudio.4dgs-converter',
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='4DGS-Converter',
        debug=False,
        strip=False,
        upx=True,
        console=False,
        icon='app/converter/icon.ico',
    )
