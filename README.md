# RK Pix - ComfyUI Custom Nodes

A collection of custom nodes for image processing, utilities, and creative effects.

## Installation

1. Clone this repository into your `ComfyUI/custom_nodes/` folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/username/comfyui-rk-pix RK_pix
```

2. Restart ComfyUI

## Nodes

### Image Processing (`rk_pix/image`)

| Node | Description |
|------|-------------|
| **Blend Images (RK)** | Blend two images with various blend modes (normal, multiply, screen, overlay, etc.) |
| **Color Adjust (RK)** | Adjust brightness, contrast, saturation, and gamma |
| **Tile Image (RK)** | Tile an image in a grid pattern |
| **Mirror Image (RK)** | Flip image horizontally, vertically, or both |
| **Crop Image (RK)** | Crop image using percentage-based coordinates |
| **Add Border (RK)** | Add a colored border around an image |
| **Advanced Image Loader (RK)** | Universal image loader with batch, frame-by-frame playback, and EXR support |
| **EXR Converter (RK)** | Dedicated EXR processor with color space conversion, tone mapping, and LUT support |

### Utility (`rk_pix/utility`)

| Node | Description |
|------|-------------|
| **Number Math (RK)** | Perform math operations (add, subtract, multiply, divide, etc.) |
| **Number Remap (RK)** | Remap a number from one range to another |
| **String Concat (RK)** | Concatenate up to 4 strings with separator |
| **String Replace (RK)** | Find and replace text in a string |
| **Image Info (RK)** | Get image dimensions, shape, and statistics |
| **Image To Mask (RK)** | Convert image to mask using luminance |
| **Mask To Image (RK)** | Convert mask to grayscale image |
| **Image Hash (RK)** | Generate MD5 hash of image for change detection |
| **Batch Selector (RK)** | Select a specific image from a batch |
| **Images To Batch (RK)** | Combine multiple images into a batch |

### Effects (`rk_pix/effect`)

| Node | Description |
|------|-------------|
| **Vignette (RK)** | Add vignette darkening effect |
| **Chromatic Aberration (RK)** | Shift RGB channels for lens distortion effect |
| **Add Noise (RK)** | Add gaussian, uniform, salt & pepper, or perlin noise |
| **Pixelate (RK)** | Reduce image resolution for pixel art effect |
| **Glitch Effect (RK)** | Add digital glitch/scanline displacement |
| **Duotone (RK)** | Convert image to two-color duotone |
| **Posterize (RK)** | Reduce color levels for poster effect |
| **Scanlines (RK)** | Add CRT monitor scanline effect |
| **Kaleidoscope (RK)** | Create kaleidoscope pattern from image |

## Usage Examples

### Blend Images
Connect two images and choose a blend mode with opacity control.

### Color Adjust
Chain multiple Color Adjust nodes for fine-tuned image correction.

### Glitch Effect
Use with a seed for reproducible glitch patterns.

### Advanced Image Loader
**Modes:**
- `single` — Load one image
- `batch_folder` — Load all images from the same folder
- `batch_pattern` — Load images matching a glob pattern (e.g., `frame_*.png`)
- `frame_by_frame` — Play one frame per workflow run (like a player)

**Frame-by-frame features:**
- Set `start_frame` and `end_frame` to define a range
- `step` controls frame skipping
- `loop` goes back to start after reaching the end
- `reset` restarts from the start frame

**Supported formats:** PNG, JPG, WEBP, BMP, TIFF, TGA, DDS, EXR, HDR, PNM, PFM, PSD, and more.

### EXR Converter
Convert EXR files with professional color management:
- **Color spaces:** Linear, sRGB, Rec.709, ACES, Custom Gamma, Custom LUT
- **Tone mapping:** None, Reinhard, ACES, Filmic, AgX
- **Output depth:** 8-bit, 16-bit, 32-bit PNG
- **Auto-save:** Save converted PNG to ComfyUI output folder
- **LUT support:** Place `.cube` files in `input/luts/` folder

## License

MIT License
