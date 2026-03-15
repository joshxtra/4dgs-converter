import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.pipeline.ply_to_h265 import quantize_position, quantize_uint8, dequantize_position, dequantize_uint8

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


if __name__ == "__main__":
    test_quantize_position_roundtrip()
    print("PASS: test_quantize_position_roundtrip")
    test_quantize_uint8_roundtrip()
    print("PASS: test_quantize_uint8_roundtrip")
    test_quantize_position_clipping()
    print("PASS: test_quantize_position_clipping")
    print("\nAll tests passed!")
