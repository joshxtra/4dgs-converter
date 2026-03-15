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

- SH degree 0 only (1 SH texture = DC color)
- Total textures per frame: 4 (position, rotation, scaleOpacity, sh_dc)
- Total attribute channels: 16

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
- Padded to multiple of 16 for codec block alignment
- Unused slots filled with zero

### Dependency

- `plas` PyTorch package from Fraunhofer HHI (https://github.com/fraunhoferhhi/PLAS)
- Requires CUDA GPU

## 2. Quantization

### Position: uint16

```python
# Per-sequence global min/max (not per-frame)
pos_norm = (position - global_min) / (global_max - global_min)  # [0, 1]
pos_u16 = (pos_norm * 65535).clip(0, 65535).astype(np.uint16)
pos_high = (pos_u16 >> 8).astype(np.uint8)   # upper 8 bits
pos_low  = (pos_u16 & 0xFF).astype(np.uint8) # lower 8 bits
```

- 65536 levels per axis
- Global min/max stored in manifest.json for dequantization
- Global (not per-frame) so video codec sees consistent value ranges

### Rotation: uint8

```python
rot_norm = (rotation - global_min) / (global_max - global_min)
rot_u8 = (rot_norm * 255).clip(0, 255).astype(np.uint8)
```

- Quaternion components, 256 levels sufficient
- May need renormalization after dequantization

### ScaleOpacity: uint8

```python
# Scale is already exp-activated, opacity is sigmoid-activated from PLY
so_norm = (scale_opacity - global_min) / (global_max - global_min)
so_u8 = (so_norm * 255).clip(0, 255).astype(np.uint8)
```

### SH DC: uint8

```python
sh_norm = (sh_dc - global_min) / (global_max - global_min)
sh_u8 = (sh_norm * 255).clip(0, 255).astype(np.uint8)
```

## 3. Channel Tiling

To avoid YUV 4:2:0 chroma subsampling destroying channel data, all channels are tiled horizontally into a single grayscale image (Y-only). Each video stream uses only the luma plane.

| Stream | Channels | Tiled resolution | Source channels |
|--------|----------|-----------------|-----------------|
| position_high | 3 (X,Y,Z high bytes) | 3264 x 1088 | position uint16 >> 8 |
| position_low | 3 (X,Y,Z low bytes) | 3264 x 1088 | position uint16 & 0xFF |
| rotation | 4 (X,Y,Z,W) | 4352 x 1088 | quaternion components |
| scale_opacity | 4 (X,Y,Z,W) | 4352 x 1088 | scale XYZ + sigmoid opacity |
| sh_dc | 4 (R,G,B,A) | 4352 x 1088 | DC color + 1 channel |

**Total: 5 video streams**

### Tiling layout

```
For 3-channel stream (e.g., position_high):
  [  X 1088px  |  Y 1088px  |  Z 1088px  ]  = 3264 x 1088

For 4-channel stream (e.g., rotation):
  [  Qx 1088px |  Qy 1088px |  Qz 1088px |  Qw 1088px  ] = 4352 x 1088
```

## 4. H.265 Encoding

### Codec settings

```bash
ffmpeg -y \
  -framerate {fps} \
  -i "temp/{stream_name}_%04d.png" \
  -c:v libx265 \
  -pix_fmt gray \
  -crf {crf} \
  -x265-params "keyint={fps}:min-keyint={fps}" \
  "output/{stream_name}.mp4"
```

### CRF per stream

| Stream | CRF | Rationale |
|--------|-----|-----------|
| position_high | 15-18 | Most sensitive, small errors cause splat jumping |
| position_low | 15-18 | Lower bits of position, equally important for precision |
| rotation | 20-24 | Medium sensitivity |
| scale_opacity | 20-24 | Medium sensitivity |
| sh_dc | 22-28 | Least sensitive, color tolerates more error |

CRF values are starting points for experimentation.

### Keyframe interval

- `keyint={fps}` = 1 keyframe per second (e.g., every 24 frames at 24fps)
- Enables segment-based streaming with 1-second granularity
- Each keyframe (IDR frame) allows random access

## 5. File Layout

```
{sequence-name}/
├── manifest.json
├── position_high.mp4
├── position_low.mp4
├── rotation.mp4
├── scale_opacity.mp4
└── sh_dc.mp4
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
      "min": [-3.0, -3.0, -3.0, -3.0],
      "max": [3.0, 3.0, 3.0, 3.0]
    }
  },

  "streams": {
    "position_high": { "file": "position_high.mp4", "tiledWidth": 3264, "channels": 3 },
    "position_low":  { "file": "position_low.mp4",  "tiledWidth": 3264, "channels": 3 },
    "rotation":      { "file": "rotation.mp4",      "tiledWidth": 4352, "channels": 4 },
    "scaleOpacity":  { "file": "scale_opacity.mp4", "tiledWidth": 4352, "channels": 4 },
    "shDC":          { "file": "sh_dc.mp4",         "tiledWidth": 4352, "channels": 4 }
  }
}
```

## 6. Browser Decode (WebGPU + WebCodecs)

### VideoDecoder setup

```javascript
const decoder = new VideoDecoder({
  output: (frame) => {
    device.queue.copyExternalImageToTexture(
      { source: frame },
      { texture: gpuTexture },
      [frame.displayWidth, frame.displayHeight]
    );
    frame.close();
  },
  error: (e) => console.error(e),
});

decoder.configure({
  codec: "hev1.1.6.L120.B0",  // H.265 Main Profile Level 4.0
  codedWidth: 3264,            // or 4352
  codedHeight: 1088,
});
```

5 decoders run in parallel, one per stream.

### Dequantize compute shader (WGSL)

```wgsl
@group(0) @binding(0) var posHighTex: texture_2d<f32>;
@group(0) @binding(1) var posLowTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> meta: QuantMeta;
@group(0) @binding(3) var outputTex: texture_storage_2d<rgba32float, write>;

struct QuantMeta {
    posMin: vec3f,
    posMax: vec3f,
}

@compute @workgroup_size(16, 16)
fn dequantizePosition(@builtin(global_invocation_id) id: vec3u) {
    let x = id.x;
    let y = id.y;
    if (x >= 1088u || y >= 1088u) { return; }

    var pos: vec3f;
    for (var ch = 0u; ch < 3u; ch++) {
        let tiled_x = i32(x + ch * 1088u);
        let high = textureLoad(posHighTex, vec2i(tiled_x, i32(y)), 0).r;
        let low  = textureLoad(posLowTex,  vec2i(tiled_x, i32(y)), 0).r;

        let u16val = u32(high * 255.0) << 8u | u32(low * 255.0);
        let norm = f32(u16val) / 65535.0;
        pos[ch] = meta.posMin[ch] + norm * (meta.posMax[ch] - meta.posMin[ch]);
    }

    textureStore(outputTex, vec2i(i32(x), i32(y)), vec4f(pos, 0.0));
}
```

### Playback buffering

- Fetch all 5 MP4 files upfront (small enough: ~50-100 MB total)
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
| RAW (uncompressed) | ~22.6 GB | ~47 MB | ~9,024 Mbps |
| GSD (shuffle+LZ4) | ~14.69 GB | ~30 MB | ~5,760 Mbps |
| H.265 (estimated) | **~50-100 MB** | **~100-200 KB** | **~19-38 Mbps** |

### Browser decode budget (per frame at 24fps = 41.7ms)

| Step | Estimated time |
|------|---------------|
| H.265 hardware decode x5 | ~2-5 ms (parallel) |
| copyExternalImageToTexture x5 | ~1-2 ms |
| Dequantize compute shader | ~1-2 ms |
| Render | ~5-10 ms |
| **Total** | **~9-19 ms** |

Fits comfortably in 41.7ms frame budget.

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Position uint16 + H.265 lossy causes splat jitter | Visual artifacts | Experiment with CRF; keep position CRF low (15-18); compare with lossless position stream |
| `pix_fmt gray` H.265 not hardware-decoded on some devices | Software decode fallback, higher CPU | Test on target devices; fallback to H.264 if needed |
| PLAS sort too slow for large gaussian counts | Long encoding time | Acceptable for offline; Inter-PLAS is faster for subsequent frames |
| Quaternion denormalization after uint8 quantize | Rotation artifacts | Renormalize in dequantize shader |
| Global min/max outliers waste quantization range | Reduced effective precision | Use percentile (0.1%-99.9%) instead of absolute min/max, clamp outliers |

## 10. Encoding Tool

### CLI interface

```bash
# From GSD
python gsd_to_h265.py \
  --input D:/4dgs-data/fish-2/fish-2.gsd \
  --output D:/4dgs-data/fish-2-encoded/ \
  --fps 24 \
  --crf-position 18 \
  --crf-rotation 22 \
  --crf-sh 24

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
3. Compute global min/max per attribute (with percentile clipping)
4. Quantize + tile → save as PNG sequence to temp/
5. ffmpeg encode 5 streams → MP4
6. Write manifest.json
7. Cleanup temp PNGs

## Future Extensions

- **SH degree 1-3**: Add more SH streams, consider PCA compression
- **Segment-based streaming**: Split MP4 into 1-second segments for DASH-like adaptive streaming
- **Adaptive bitrate**: Encode at multiple CRF levels, client selects based on bandwidth
- **LOD / progressive**: Encode subset of gaussians for quick initial load, then full set
