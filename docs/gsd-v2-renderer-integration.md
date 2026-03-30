# GSD V2 Renderer Integration Guide

## Overview

GSD V2 adds a new compression type `"sharp_vq"` that uses per-frame vector quantization for rotation, scale, and SH DC attributes. The file structure (magic + JSON header + sequential frame blobs) is unchanged from V1.

The renderer detects V2 via the header field `"compression": "sharp_vq"` (V1 uses `"shuffle_lz4"`).

## Header Changes

V2 header JSON adds these fields:

```json
{
  "version": 2,
  "compression": "sharp_vq",
  "rotationEncoding": "vq256",
  "scaleEncoding": "vq256",
  "opacityEncoding": "uint8",
  "shEncoding": "vq256",
  "positionEncoding": "fp16_shuffle_lz4",
  // ... existing fields unchanged: frameCount, targetFPS, textureWidth, etc.
}
```

Position precision is always fp16 in V2 (no fp32 option). The `positionPrecision`, `rotationPrecision`, `scaleOpacityPrecision`, `shPrecision` fields are still present for metadata but the actual encoding is determined by the `*Encoding` fields.

## Per-Frame Blob Structure

Each frame blob (after the 4-byte `compressedSize` prefix) contains:

```
[20 bytes] Section sizes header: 5 × uint32 LE
  - pos_size
  - rot_size
  - scale_size
  - opacity_size
  - sh_size

[pos_size bytes]     LZ4-compressed position data (byte-shuffled fp16×3)
[rot_size bytes]     LZ4-compressed rotation VQ indices (uint8)
[scale_size bytes]   LZ4-compressed scale VQ indices (uint8)
[opacity_size bytes] LZ4-compressed opacity data (byte-shuffled uint8)
[sh_size bytes]      LZ4-compressed SH DC VQ indices (uint8)
```

## Per-Frame Codebook Storage

Each frame has its own codebooks stored BEFORE the section data:

```
[20 bytes]           Section sizes (as above)
[3072 bytes]         Rotation codebook: 256 × float32 × 4 = 4096 bytes
[3072 bytes]         Scale codebook: 256 × float32 × 3 = 3072 bytes
[3072 bytes]         SH DC codebook: 256 × float32 × 3 = 3072 bytes
[pos_size bytes]     ... sections as above
```

Updated section sizes header (7 × uint32):
```
[28 bytes] Section sizes: 7 × uint32 LE
  - rot_codebook_size (always 4096: 256 entries × 4 floats × 4 bytes)
  - scale_codebook_size (always 3072: 256 entries × 3 floats × 4 bytes)
  - sh_codebook_size (always 3072: 256 entries × 3 floats × 4 bytes)
  - pos_size
  - rot_size
  - scale_size
  - opacity_size
  - sh_size
```

Note: codebooks are NOT LZ4-compressed (they're small and random, compression doesn't help). They're stored as raw float32 arrays.

## Decode Pipeline

### Step 1: Read and LZ4-decompress each section

Same as V1 but sections are different:

```cpp
// Read section sizes
uint32 Sizes[8];
FMemory::Memcpy(Sizes, CompressedData, 32);

uint32 RotCBSize = Sizes[0];   // 4096
uint32 ScaleCBSize = Sizes[1]; // 3072
uint32 ShCBSize = Sizes[2];    // 3072
uint32 PosSize = Sizes[3];
uint32 RotSize = Sizes[4];
uint32 ScaleSize = Sizes[5];
uint32 OpacitySize = Sizes[6];
uint32 ShSize = Sizes[7];
```

### Step 2: Position (same as V1)

```cpp
// LZ4 decompress → byte unshuffle → fp16 RGBA-ish (actually RGB, 6 bpp)
// Note: V2 position is fp16×3 (6 bytes/pixel), NOT fp16×4 (8 bytes/pixel)
// Pack into PF_FloatRGBA texture with A=0, or use a 3-channel format
```

**Important change:** V2 position is 3-channel fp16 (6 bpp), not 4-channel. On GPU, load into a `PF_FloatRGBA` texture with alpha = 0. The byte-unshuffle BPP is 6, not 8.

Actually, for simplicity, the encoder can pad to 4-channel fp16 (8 bpp) to keep compatibility with the existing `PF_FloatRGBA` texture. This is a design decision — padding wastes ~1 MB but simplifies the renderer. **Recommendation: pad to 4-channel for now.**

### Step 3: VQ attributes (NEW)

For rotation, scale, and SH DC, the decoder has two options:

#### Option A: CPU decode (simpler, recommended for first implementation)

```cpp
// Load codebook (raw float32 array)
const float* RotCodebook = (const float*)(Data + RotCBOffset); // 256 × 4 floats

// LZ4 decompress indices
TArray<uint8> RotIndices;
RotIndices.SetNumUninitialized(NumPixels);
FCompression::UncompressMemory(NAME_LZ4, RotIndices.GetData(), NumPixels,
                               CompressedRotData, RotCompressedSize);

// Codebook lookup → produce fp16 RGBA texture
TArray<uint8> RotTextureData;
RotTextureData.SetNumUninitialized(NumPixels * 8); // fp16 RGBA = 8 bpp
FFloat16* RotOut = (FFloat16*)RotTextureData.GetData();

for (int32 i = 0; i < NumPixels; i++)
{
    uint8 Idx = RotIndices[i];
    const float* Entry = RotCodebook + Idx * 4; // 4 floats per entry
    RotOut[i * 4 + 0] = FFloat16(Entry[0]);
    RotOut[i * 4 + 1] = FFloat16(Entry[1]);
    RotOut[i * 4 + 2] = FFloat16(Entry[2]);
    RotOut[i * 4 + 3] = FFloat16(Entry[3]);
}
// Upload RotTextureData as PF_FloatRGBA texture (same as V1)
```

Scale and SH DC are similar but 3-channel (pack into RGBA with A=0 or A=1).

#### Option B: GPU decode (faster, requires shader changes)

Upload codebook as a `StructuredBuffer` and indices as a `R8_UINT` texture. Decode in the compute shader:

```hlsl
// New resources
StructuredBuffer<float4> RotCodebook;     // 256 entries
Texture2D<uint> RotIndexTexture;          // R8_UINT

StructuredBuffer<float3> ScaleCodebook;   // 256 entries
Texture2D<uint> ScaleIndexTexture;

StructuredBuffer<float3> SHCodebook;      // 256 entries
Texture2D<uint> SHIndexTexture;

// In CSGaussianCommon.ush, add new read functions:
float4 ReadRotationVQ(Texture2D<uint> indexTex, StructuredBuffer<float4> codebook,
                      uint idx, uint tex_width)
{
    uint2 tc = uint2(idx % tex_width, idx / tex_width);
    uint cbIdx = indexTex.Load(int3(tc, 0));
    return codebook[cbIdx];
}

float3 ReadScaleVQ(Texture2D<uint> indexTex, StructuredBuffer<float3> codebook,
                   uint idx, uint tex_width)
{
    uint2 tc = uint2(idx % tex_width, idx / tex_width);
    uint cbIdx = indexTex.Load(int3(tc, 0));
    return codebook[cbIdx];
}
```

Then in `CSComputeTransform.usf`:

```hlsl
// V1 path (existing)
float4 quat = ReadRotation(RotationTexture, idx, texture_width);
float3 scale = ReadScale(ScaleOpacityTexture, idx, texture_width);

// V2 path (new)
float4 quat = ReadRotationVQ(RotIndexTexture, RotCodebook, idx, texture_width);
float3 scale = ReadScaleVQ(ScaleIndexTexture, ScaleCodebook, idx, texture_width);
```

### Step 4: Opacity

```cpp
// LZ4 decompress → uint8 array
TArray<uint8> OpacityU8;
OpacityU8.SetNumUninitialized(NumPixels);
FCompression::UncompressMemory(NAME_LZ4, OpacityU8.GetData(), NumPixels, ...);

// CPU decode to fp16 for texture upload:
for (int32 i = 0; i < NumPixels; i++)
{
    float Val = OpacityU8[i] / 255.0f;
    // Pack into the alpha channel of ScaleOpacity texture,
    // or into a separate opacity texture
}

// OR GPU decode: upload as R8_UNORM texture, shader reads directly
// GPU auto-maps [0,255] → [0.0, 1.0]
```

## Files to Modify in SplatRenderer

### Minimum changes (Option A: CPU decode)

1. **CSGSDReader.h** — Add V2 detection flag, codebook storage
2. **CSGSDReader.cpp** — Parse V2 header, decode VQ in `ReadFrame()`
3. **CSGaussianSequencePlayer.cpp** — Handle V2 frame data (textures are produced by CSGSDReader, so minimal changes)

### For GPU decode (Option B)

Additional changes:

4. **CSGaussianShaders.h** — Add `StructuredBuffer` and `Texture2D<uint>` parameters
5. **CSGaussianRendering.cpp** — Bind codebook buffers and index textures
6. **CSGaussianCommon.ush** — Add `ReadRotationVQ`, `ReadScaleVQ`, `ReadSHVQ`
7. **CSComputeTransform.usf** — Use VQ read functions when V2
8. **CSRenderSplatVS.usf** — Read opacity from uint8 texture or decoded texture

## Recommended Implementation Order

1. Start with **Option A (CPU decode)** — no shader changes, just modify CSGSDReader
2. Verify visual quality matches V1
3. Profile performance
4. If needed, migrate to **Option B (GPU decode)** for speed

## Backward Compatibility

- V1 GSD files (`"compression": "shuffle_lz4"`) work exactly as before
- CSGSDReader checks `compression` field and branches:
  - `"shuffle_lz4"` → existing V1 path
  - `"sharp_vq"` → new V2 path

## Data Size Comparison

| | V1 (shuffle_lz4) | V2 (sharp_vq) |
|---|---|---|
| Per frame | ~30 MB | ~8.5 MB |
| 480 frames | ~15 GB | ~4.1 GB |
| Decode speed | ~15ms | ~5-8ms (estimated) |

## Quality Impact

| Attribute | Encoding | Error |
|---|---|---|
| Position | fp16 (lossless vs V1) | 0 |
| Rotation | VQ K=256 | ~0.9° mean |
| Scale | VQ K=256 | 0.065 mean abs |
| Opacity | uint8 | 0.002 mean abs |
| SH DC | VQ K=256 | 0.017 mean abs |

All errors are within visually acceptable range based on testing across 5 SHARP scenes (fish-2, camel, elephant, city_night, landscape_sunset_sea).
