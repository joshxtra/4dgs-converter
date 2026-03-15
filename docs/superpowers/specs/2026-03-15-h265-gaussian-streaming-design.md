# H.265 Gaussian Splatting Streaming Format

**Date:** 2026-03-15
**Status:** Draft
**Goal:** Replace GSD format with H.265-encoded video streams for web (WebGPU) and UE5 playback, achieving 17-75 Mbps streaming bitrate.

## Background

Current GSD format stores per-frame gaussian data with Byte-Shuffle + LZ4 compression, yielding ~30 MB/frame (~5,760 Mbps at 24fps). This is viable for local disk playback but impossible for internet streaming. Production systems like Gracia.ai stream at 17-75 Mbps, implying 150-300x better compression via video codec inter-frame prediction.

### Research Basis

- **PLAS** (Fraunhofer HHI, ECCV '24): Sorts gaussians into spatially coherent 2D grids
- **GSCV** (OpenReview): Inter-PLAS for temporal consistency across frames
- **V3** (arxiv 2409.13648): Streams 2D gaussians to mobile via H.264, uint16 position + uint8 attributes
- **TF4DGC** (IEEE): Standard H.264/H.265 codec with UV mapping, hardware decoder compatible
- **GIFStream**: 30 Mbps immersive video with real-time rendering

## Architecture Overview

```
Encoding (offline, Python + ffmpeg):
  PLY frames → PLAS sort → quantize → tile channels → H.265 encode → MP4 files

Playback (browser):
  fetch MP4 → WebCodecs VideoDecoder → VideoFrame → WebGPU texture → compute shader dequantize → render

Playback (UE5, future):
  MP4 → UMediaPlayer (hardware decode) → UMediaTexture → compute shader dequantize → Niagara render
```

Same MP4 files serve both WebGPU and UE5.

## Scope

- SH degree 0 only (1 SH texture = DC color, 3 channels RGB)
- Total textures per frame: 4 (position, rotation, scaleOpacity, sh_dc)
- Source attribute channels: 14 (position 3 + rotation 4 + scaleOpacity 4 + shDC 3)
- Stream channels after uint16 position split: 17 (posHigh 3 + posLow 3 + rotation 4 + scaleOpacity 4 + shDC 3)

## 1. Spatial Sorting: PLAS + Inter-PLAS

### Why sorting is needed

Video codecs assume adjacent pixels are correlated. Raw gaussian texture packing has arbitrary ordering — adjacent pixels are unrelated gaussians. Without sorting, H.265 intra/inter prediction is ineffective.

### Algorithm

- **Frame 0**: Full PLAS sort (~60s GPU for 1.2M gaussians)
  - Input: 6D vectors (position XYZ + SH DC RGB) per gaussian
  - Output: permutation mapping gaussians to 1088x1088 grid positions
  - Minimizes L2 distance between spatial neighbors
- **Frame 1+**: Inter-PLAS
  - Initialize with previous frame's permutation
  - Local optimization only — maintains temporal consistency
  - Critical for H.265 inter-frame compression (motion estimation works when same grid position corresponds to similar gaussian across frames)

### Grid dimensions

- 1088 x 1088 = 1,183,744 slots (>= 1,179,648 gaussians)
- 1088 = 1087 rounded up to nearest multiple of 16 for codec block alignment
- Unused slots filled with zero

### Dependency

- `plas` PyTorch package from Fraunhofer HHI (https://github.com/fraunhoferhhi/PLAS)
- Requires CUDA GPU

## 2. Quantization

All quantization uses **global min/max across the entire sequence** (not per-frame) so the video codec sees consistent value ranges. Use percentile clipping (0.1%-99.9%) to avoid outliers wasting quantization range.

### Position: uint16

```python
# Per-sequence global min/max with percentile clipping
pos_min = np.percentile(all_positions, 0.1, axis=0)
pos_max = np.percentile(all_positions, 99.9, axis=0)
pos_norm = (position - pos_min) / (pos_max - pos_min)  # [0, 1]
pos_u16 = (pos_norm * 65535).clip(0, 65535).astype(np.uint16)
pos_high = (pos_u16 >> 8).astype(np.uint8)   # upper 8 bits
pos_low  = (pos_u16 & 0xFF).astype(np.uint8) # lower 8 bits
```

- 65536 levels per axis
- Global min/max stored in manifest.json for dequantization

### Rotation: uint8

```python
rot_norm = (rotation - rot_min) / (rot_max - rot_min)
rot_u8 = (rot_norm * 255).clip(0, 255).astype(np.uint8)
```

- Quaternion components, 256 levels per component
- After dequantization, quaternion must be renormalized
- **Risk**: uint8 gives ~0.5 degree angular precision. Combined with H.265 lossy compression, effective precision may drop to ~1-2 degrees. Monitor rotation quality during CRF experiments — if artifacts visible, consider upgrading to uint16 (like position)

### ScaleOpacity: uint8

```python
# Scale is already exp-activated, opacity is sigmoid-activated from PLY
so_norm = (scale_opacity - so_min) / (so_max - so_min)
so_u8 = (so_norm * 255).clip(0, 255).astype(np.uint8)
```

### SH DC: uint8

```python
sh_norm = (sh_dc - sh_min) / (sh_max - sh_min)
sh_u8 = (sh_norm * 255).clip(0, 255).astype(np.uint8)
```

### Coordinate system

Quantization operates on the raw attribute values from PLY (COLMAP coordinate space). The COLMAP-to-rendering coordinate transform (e.g., axis swaps, scaling) happens in the dequantize compute shader at playback time, not during encoding. This keeps the encoding pipeline coordinate-system-agnostic.

## 3. Channel Tiling

To avoid YUV 4:2:0 chroma subsampling destroying channel data, multiple attribute channels are tiled side-by-side into a single image. The data is placed only in the Y (luma) plane; U/V planes are filled with 128 (neutral) and ignored. This keeps the bitstream within **H.265 Main Profile** for universal hardware decode support.

### Stream layout: 3 streams

To stay within H.265 **Level 5.0** limits (max 8,912,896 luma samples per picture) and avoid exceeding hardware decoder session limits (typically 2-4), all channels are consolidated into **3 video streams**:

| Stream | Contents | Channels | Tiled layout | Resolution | Pixels |
|--------|----------|----------|-------------|-----------|--------|
| stream_position | posHigh XYZ + posLow XYZ | 6 | 3 cols × 2 rows | 3264 × 2176 | 7,102,464 |
| stream_motion | rotation XYZW + scaleOpacity XY | 6 | 3 cols × 2 rows | 3264 × 2176 | 7,102,464 |
| stream_appearance | scaleOpacity ZW + shDC RGB | 5 | 5 cols × 1 row | 5440 × 1088 | 5,918,720 |

All streams are under 8,912,896 pixels → **Level 5.0 compatible**.
3 decoder sessions → within hardware limits of all modern GPUs.

### Tiling detail

```
stream_position (3264 × 2176):
  Row 0: [ posHi_X 1088px | posHi_Y 1088px | posHi_Z 1088px ]
  Row 1: [ posLo_X 1088px | posLo_Y 1088px | posLo_Z 1088px ]

stream_motion (3264 × 2176):
  Row 0: [ rot_X 1088px | rot_Y 1088px | rot_Z 1088px ]
  Row 1: [ rot_W 1088px | so_X  1088px | so_Y  1088px ]

stream_appearance (5440 × 1088):
  Row 0: [ so_Z 1088px | so_W 1088px | shDC_R 1088px | shDC_G 1088px | shDC_B 1088px ]
```

### CRF per stream

| Stream | CRF | Rationale |
|--------|-----|-----------|
| stream_position | 15-18 | Most sensitive — position errors cause splat jumping. Note: posLow bytes are less sensitive than posHigh but share the same stream |
| stream_motion | 20-24 | Medium sensitivity — rotation + scale |
| stream_appearance | 22-28 | Least sensitive — opacity + DC color tolerate more error |

CRF values are starting points for experimentation.

## 4. H.265 Encoding

### Codec settings

```bash
ffmpeg -y \
  -framerate {fps} \
  -i "temp/{stream_name}_%04d.png" \
  -c:v libx265 \
  -pix_fmt yuv420p \
  -crf {crf} \
  -x265-params "keyint={fps}:min-keyint={fps}" \
  "output/{stream_name}.mp4"
```

Key choices:
- `pix_fmt yuv420p`: Ensures **Main Profile** bitstream for universal hardware decode. Data lives in Y plane only; U/V planes are neutral (128) and wasted (~25% overhead, acceptable)
- `keyint={fps}`: 1 keyframe (IDR) per second for random access and segment-based streaming

### Alternative: rawvideo pipe (faster encoding)

To avoid PNG intermediate files:
```bash
ffmpeg -y \
  -f rawvideo -pix_fmt gray -s {width}x{height} -framerate {fps} \
  -i pipe:0 \
  -c:v libx265 \
  -pix_fmt yuv420p \
  -crf {crf} \
  -x265-params "keyint={fps}:min-keyint={fps}" \
  "output/{stream_name}.mp4"
```
Feed quantized frame bytes directly via stdin. Eliminates PNG encode/decode overhead and disk I/O for intermediate files.

## 5. File Layout

```
{sequence-name}/
├── manifest.json
├── stream_position.mp4
├── stream_motion.mp4
└── stream_appearance.mp4
```

### manifest.json

```json
{
  "version": 1,
  "format": "4dgs-h265",
  "sequenceName": "fish-2",
  "frameCount": 480,
  "targetFPS": 24,
  "duration": 20.0,
  "shDegree": 0,
  "gridWidth": 1088,
  "gridHeight": 1088,
  "gaussianCount": 1179648,

  "requiredCodec": "hev1.1.6.L150.B0",
  "coordinateSpace": "colmap",
  "quaternionOrder": "wxyz",

  "quantization": {
    "position": {
      "precision": "uint16",
      "min": [-2.31, -1.85, -3.02],
      "max": [2.45, 1.92, 2.88]
    },
    "rotation": {
      "precision": "uint8",
      "min": [-1.0, -1.0, -1.0, -1.0],
      "max": [1.0, 1.0, 1.0, 1.0]
    },
    "scaleOpacity": {
      "precision": "uint8",
      "min": [0.0001, 0.0001, 0.0001, 0.0],
      "max": [0.5, 0.5, 0.5, 1.0]
    },
    "shDC": {
      "precision": "uint8",
      "min": [-3.0, -3.0, -3.0],
      "max": [3.0, 3.0, 3.0]
    }
  },

  "streams": {
    "position":   { "file": "stream_position.mp4",   "width": 3264, "height": 2176, "channels": 6 },
    "motion":     { "file": "stream_motion.mp4",     "width": 3264, "height": 2176, "channels": 6 },
    "appearance": { "file": "stream_appearance.mp4", "width": 5440, "height": 1088, "channels": 5 }
  }
}
```

## 6. Browser Decode (WebGPU + WebCodecs)

### Codec capability check

```javascript
const supported = await VideoDecoder.isConfigSupported({
  codec: "hev1.1.6.L150.B0",
  codedWidth: 5440,
  codedHeight: 1088,
});
if (!supported.supported) {
  // fallback to H.264 or show error
}
```

### VideoDecoder setup

```javascript
// 3 decoders, one per stream
for (const [name, streamInfo] of Object.entries(manifest.streams)) {
  const decoder = new VideoDecoder({
    output: (frame) => {
      device.queue.copyExternalImageToTexture(
        { source: frame },
        { texture: streamTextures[name] },
        [frame.displayWidth, frame.displayHeight]
      );
      frame.close();
    },
    error: (e) => console.error(e),
  });

  decoder.configure({
    codec: manifest.requiredCodec,  // "hev1.1.6.L150.B0"
    codedWidth: streamInfo.width,
    codedHeight: streamInfo.height,
  });
}
```

### Frame synchronization

All 3 decoders must produce frame N before compositing. Strategy:
- Track decoded frame index per decoder via output callback
- Only dispatch render when all 3 streams have frame N ready
- If one decoder falls behind, wait (don't render mixed frames)
- On persistent desync, seek all decoders to next keyframe

### Dequantize compute shader (WGSL)

```wgsl
@group(0) @binding(0) var positionTex: texture_2d<f32>;  // stream_position: 3264x2176
@group(0) @binding(1) var<uniform> meta: QuantMeta;
@group(0) @binding(2) var outputPos: texture_storage_2d<rgba32float, write>;

struct QuantMeta {
    posMin: vec3f,
    _pad0: f32,
    posMax: vec3f,
    _pad1: f32,
}

@compute @workgroup_size(16, 16)
fn dequantizePosition(@builtin(global_invocation_id) id: vec3u) {
    let x = id.x;
    let y = id.y;
    if (x >= 1088u || y >= 1088u) { return; }

    var pos: vec3f;
    for (var ch = 0u; ch < 3u; ch++) {
        let tiled_x = i32(x + ch * 1088u);
        // Row 0 = high bytes, Row 1 = low bytes
        let high = textureLoad(positionTex, vec2i(tiled_x, i32(y)), 0).r;
        let low  = textureLoad(positionTex, vec2i(tiled_x, i32(y + 1088u)), 0).r;

        // Reconstruct uint16 with rounding correction
        let u16val = u32(high * 255.0 + 0.5) << 8u | u32(low * 255.0 + 0.5);
        let norm = f32(u16val) / 65535.0;
        pos[ch] = meta.posMin[ch] + norm * (meta.posMax[ch] - meta.posMin[ch]);
    }

    textureStore(outputPos, vec2i(i32(x), i32(y)), vec4f(pos, 0.0));
}
```

Note: `+ 0.5` before `u32()` cast prevents off-by-one from float truncation.

### Playback buffering

- Fetch all 3 MP4 files upfront (small enough: ~50-100 MB total)
- Or fetch segment-by-segment for longer sequences
- Decode 2-3 frames ahead into ring buffer of WebGPU textures
- Render from ring buffer at target FPS

## 7. UE5 Playback (Future)

Same MP4 files. Replace current GSD decode path in GSRawStream with:

1. `UMediaPlayer` loads MP4 → hardware H.265 decode
2. `UMediaTexture` provides decoded frame as UE texture
3. Compute shader dequantizes (same logic as WGSL, ported to HLSL)
4. Feed into existing Niagara rendering pipeline

## 8. Estimated Performance

### File sizes (fish-2, 480 frames, 24fps, 20s)

| Format | Total size | Per-frame | Bitrate |
|--------|-----------|-----------|---------|
| RAW (uncompressed, SH0) | ~22.6 GB | ~47 MB | ~9,024 Mbps |
| GSD (shuffle+LZ4) | ~14.69 GB | ~30 MB | ~5,760 Mbps |
| H.265 (estimated) | **~50-100 MB** | **~100-200 KB** | **~19-38 Mbps** |

These estimates should be validated with a small-scale experiment (encode 10-20 frames, measure actual bitrate) before committing to the architecture.

### Browser decode budget (per frame at 24fps = 41.7ms)

| Step | Estimated time |
|------|---------------|
| H.265 hardware decode ×3 | ~2-5 ms (parallel) |
| copyExternalImageToTexture ×3 | ~1-2 ms |
| Dequantize compute shader | ~1-2 ms |
| Render | ~5-10 ms |
| **Total** | **~9-19 ms** |

Fits comfortably in 41.7ms frame budget.

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Position uint16 + H.265 lossy causes splat jitter | Visual artifacts | Experiment with CRF 15-18; compare with lossless position stream |
| Rotation uint8 too coarse (~0.5° precision before codec, ~1-2° after) | Rotation artifacts on small splats | Monitor quality; upgrade to uint16 if needed |
| YUV420 U/V overhead wastes ~25% bitrate | Slightly larger files | Acceptable trade-off for guaranteed hardware decode |
| PLAS sort too slow for large gaussian counts | Long encoding time | Acceptable for offline; Inter-PLAS is faster for subsequent frames |
| Quaternion denormalization after uint8 quantize | Rotation artifacts | Renormalize in dequantize shader |
| Global min/max outliers waste quantization range | Reduced effective precision | Use percentile clipping (0.1%-99.9%) |
| Multi-stream frame desync in browser | Mixed-frame visual glitch | Barrier sync: only render when all 3 decoders have same frame |

## 10. Encoding Tool

### CLI interface

```bash
# From GSD
python gsd_to_h265.py \
  --input D:/4dgs-data/fish-2/fish-2.gsd \
  --output D:/4dgs-data/fish-2-encoded/ \
  --fps 24 \
  --crf-position 18 \
  --crf-motion 22 \
  --crf-appearance 24

# From PLY sequence
python ply_to_h265.py \
  --input D:/4dgs-data/fish-2/plys/ \
  --output D:/4dgs-data/fish-2-encoded/ \
  --fps 24
```

### Dependencies

- Python 3.10+
- PyTorch + CUDA (for PLAS)
- `plas` package (`pip install plas`)
- numpy, lz4 (for GSD reading)
- ffmpeg in system PATH

### Pipeline steps

1. Load all frames (GSD decompress or PLY parse)
2. PLAS sort frame 0, Inter-PLAS frame 1+
3. Compute global min/max per attribute (with 0.1%-99.9% percentile clipping)
4. Quantize all frames
5. Tile channels into stream images
6. Pipe to ffmpeg via rawvideo stdin → 3 × MP4
7. Write manifest.json

## Future Extensions

- **SH degree 1-3**: Add more streams, consider PCA to reduce SH channels
- **Segment-based streaming**: Split MP4 into 1-second segments for DASH-like adaptive streaming
- **Adaptive bitrate**: Encode at multiple CRF levels, client selects based on bandwidth
- **LOD / progressive**: Encode subset of gaussians for quick initial load, then full set
- **Rotation uint16**: If uint8 proves too coarse, upgrade to uint16 split like position
