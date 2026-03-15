import numpy as np
import sys
import os
import tempfile
import subprocess
import json
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.pipeline.ply_to_h265 import quantize_position, quantize_uint8, dequantize_position, dequantize_uint8, convert_ply_to_h265
from app.pipeline.ply_to_h265 import tile_stream_position, tile_stream_motion, tile_stream_appearance
from app.pipeline.ply_to_h265 import start_encoder, write_frame, finish_encoder
from app.pipeline.ply_to_h265 import write_manifest

def test_quantize_position_roundtrip():
    """uint16 quantization should have max error < 1/65535 of range."""
    pos = np.array([[1.5, -0.3, 2.1], [0.0, 0.0, 0.0], [-1.2, 1.8, -0.5]], dtype=np.float32)
    bounds_min = np.array([-2.0, -2.0, -2.0])
    bounds_max = np.array([3.0, 3.0, 3.0])

    high, low = quantize_position(pos, bounds_min, bounds_max)
    assert high.dtype == np.uint8
    assert low.dtype == np.uint8
    assert high.shape == (3, 3)
    assert low.shape == (3, 3)

    reconstructed = dequantize_position(high, low, bounds_min, bounds_max)
    max_err = np.max(np.abs(reconstructed - pos))
    assert max_err < (5.0 / 65535) * 2  # within 2 quantization steps

def test_quantize_uint8_roundtrip():
    """uint8 quantization should have max error < 1/255 of range."""
    data = np.array([[0.5, -0.8, 0.2, 1.0], [0.0, 0.3, -1.0, 0.7]], dtype=np.float32)
    d_min = np.array([-1.0, -1.0, -1.0, -1.0])
    d_max = np.array([1.0, 1.0, 1.0, 1.0])

    q = quantize_uint8(data, d_min, d_max)
    assert q.dtype == np.uint8
    assert q.shape == (2, 4)

    reconstructed = dequantize_uint8(q, d_min, d_max)
    max_err = np.max(np.abs(reconstructed - data))
    assert max_err < (2.0 / 255) * 2

def test_quantize_position_clipping():
    """Values outside bounds should be clipped to 0 or 65535."""
    pos = np.array([[10.0, -10.0, 0.0]], dtype=np.float32)
    bounds_min = np.array([-1.0, -1.0, -1.0])
    bounds_max = np.array([1.0, 1.0, 1.0])

    high, low = quantize_position(pos, bounds_min, bounds_max)
    assert high[0, 0] == 255 and low[0, 0] == 255  # clipped to max
    assert high[0, 1] == 0 and low[0, 1] == 0      # clipped to min


def test_tile_stream_position_shape():
    grid_h, grid_w = 16, 16
    n = grid_h * grid_w
    pos_high = np.arange(n * 3, dtype=np.uint8).reshape(n, 3)
    pos_low = np.arange(n * 3, dtype=np.uint8).reshape(n, 3)
    result = tile_stream_position(pos_high, pos_low, grid_h, grid_w)
    assert result.shape == (grid_h * 2, grid_w * 3)
    assert result.dtype == np.uint8

def test_tile_stream_position_layout():
    grid_h, grid_w = 4, 4
    n = 16
    pos_high = np.zeros((n, 3), dtype=np.uint8)
    pos_high[0, 0] = 10
    pos_high[0, 1] = 20
    pos_high[0, 2] = 30
    pos_low = np.zeros((n, 3), dtype=np.uint8)
    pos_low[0, 0] = 11
    result = tile_stream_position(pos_high, pos_low, grid_h, grid_w)
    assert result[0, 0] == 10
    assert result[0, grid_w] == 20
    assert result[0, grid_w*2] == 30
    assert result[grid_h, 0] == 11

def test_tile_stream_motion_shape():
    grid_h, grid_w = 16, 16
    n = grid_h * grid_w
    rot = np.zeros((n, 4), dtype=np.uint8)
    so = np.zeros((n, 4), dtype=np.uint8)
    result = tile_stream_motion(rot, so, grid_h, grid_w)
    assert result.shape == (grid_h * 2, grid_w * 3)

def test_tile_stream_appearance_shape():
    grid_h, grid_w = 16, 16
    n = grid_h * grid_w
    so_zw = np.zeros((n, 2), dtype=np.uint8)
    sh_dc = np.zeros((n, 3), dtype=np.uint8)
    result = tile_stream_appearance(so_zw, sh_dc, grid_h, grid_w)
    assert result.shape == (grid_h, grid_w * 5)

def test_ffmpeg_encode_produces_valid_mp4():
    """Encode 5 synthetic grayscale frames and verify output is valid MP4."""
    width, height, fps = 48, 32, 24
    frames = [np.random.randint(0, 256, (height, width), dtype=np.uint8) for _ in range(5)]

    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "test.mp4")
        proc = start_encoder(width, height, fps, crf=23, output_path=out_path)
        for f in frames:
            write_frame(proc, f)
        finish_encoder(proc)

        assert Path(out_path).exists()
        assert Path(out_path).stat().st_size > 0

        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=width,height,codec_name",
             "-of", "json", out_path],
            capture_output=True, text=True
        )
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        assert stream["codec_name"] == "hevc"
        assert int(stream["width"]) == width
        assert int(stream["height"]) == height

def test_manifest_structure():
    bounds = {
        "position": {"min": np.array([-1, -1, -1], dtype=np.float32),
                      "max": np.array([1, 1, 1], dtype=np.float32)},
        "rotation": {"min": np.array([-1, -1, -1, -1], dtype=np.float32),
                      "max": np.array([1, 1, 1, 1], dtype=np.float32)},
        "scale_opacity": {"min": np.array([0, 0, 0, 0], dtype=np.float32),
                           "max": np.array([1, 1, 1, 1], dtype=np.float32)},
        "sh_dc": {"min": np.array([-1, -1, -1], dtype=np.float32),
                   "max": np.array([1, 1, 1], dtype=np.float32)},
    }

    with tempfile.TemporaryDirectory() as tmp:
        write_manifest(
            output_dir=tmp,
            sequence_name="test-seq",
            frame_count=10,
            fps=24,
            grid_w=1088, grid_h=1088,
            gaussian_count=1179648,
            bounds=bounds,
        )

        manifest_path = Path(tmp) / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            m = json.load(f)

        assert m["format"] == "4dgs-h265"
        assert m["frameCount"] == 10
        assert m["targetFPS"] == 24
        assert m["shDegree"] == 0
        assert m["coordinateSpace"] == "colmap"
        assert m["quaternionOrder"] == "wxyz"
        assert "position" in m["quantization"]
        assert len(m["streams"]) == 3
        assert "position" in m["streams"]
        assert "motion" in m["streams"]
        assert "appearance" in m["streams"]


def test_ply_to_h265_integration():
    """Full pipeline: PLY frames → 3 MP4s + manifest.json."""
    ply_dir = Path("ref/io-example/step3_image-to-ply/fish-2_ply")
    if not ply_dir.exists():
        import pytest
        pytest.skip("Reference PLY data not available")

    with tempfile.TemporaryDirectory() as tmp:
        result = convert_ply_to_h265(
            ply_dir=str(ply_dir),
            output_dir=tmp,
            fps=24,
            crf_position=20,
            crf_motion=24,
            crf_appearance=26,
        )

        out = Path(tmp)
        assert (out / "stream_position.mp4").exists()
        assert (out / "stream_motion.mp4").exists()
        assert (out / "stream_appearance.mp4").exists()

        with open(out / "manifest.json") as f:
            m = json.load(f)
        assert m["frameCount"] == result["frame_count"]
        assert m["gridWidth"] == result["grid_size"]

        # MP4 files should be much smaller than raw data
        total_mp4 = sum(f.stat().st_size for f in out.glob("*.mp4"))
        # Raw per gaussian SH0: pos(12B) + rot(16B) + so(16B) + sh_dc(12B) = 56 bytes (fp32)
        raw_estimate = result["gaussian_count"] * 56 * result["frame_count"]
        compression_ratio = raw_estimate / total_mp4
        print(f"\nCompression ratio: {compression_ratio:.0f}x")
        assert compression_ratio > 5


if __name__ == "__main__":
    test_quantize_position_roundtrip()
    print("PASS: test_quantize_position_roundtrip")
    test_quantize_uint8_roundtrip()
    print("PASS: test_quantize_uint8_roundtrip")
    test_quantize_position_clipping()
    print("PASS: test_quantize_position_clipping")
    test_tile_stream_position_shape()
    print("PASS: test_tile_stream_position_shape")
    test_tile_stream_position_layout()
    print("PASS: test_tile_stream_position_layout")
    test_tile_stream_motion_shape()
    print("PASS: test_tile_stream_motion_shape")
    test_tile_stream_appearance_shape()
    print("PASS: test_tile_stream_appearance_shape")
    test_ffmpeg_encode_produces_valid_mp4()
    print("PASS: test_ffmpeg_encode_produces_valid_mp4")
    test_manifest_structure()
    print("PASS: test_manifest_structure")
    print("\nAll tests passed!")
