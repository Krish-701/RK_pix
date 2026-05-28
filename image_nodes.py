import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import comfy.utils
import os
import folder_paths
import glob
import struct
import io
import hashlib

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import OpenEXR
    import Imath
    OPENEXR_AVAILABLE = True
except ImportError:
    OPENEXR_AVAILABLE = False

# ============================================================
# IMAGE PROCESSING NODES
# ============================================================

class RKImageBlend:
    """Blend two images with various modes."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "hard_light", "difference", "add", "subtract"], {"default": "normal"}),
                "opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_image",)
    CATEGORY = "rk_pix/image"
    FUNCTION = "blend"
    
    def blend(self, image1, image2, blend_mode, opacity):
        # Ensure same size
        h1, w1 = image1.shape[1:3]
        h2, w2 = image2.shape[1:3]
        
        if h1 != h2 or w1 != w2:
            # Resize image2 to match image1
            image2 = comfy.utils.common_upscale(
                image2.movedim(-1, 1), w1, h1, "bilinear", "center"
            ).movedim(1, -1)
        
        result = image1.clone()
        
        if blend_mode == "normal":
            result = image1 * (1 - opacity) + image2 * opacity
        elif blend_mode == "multiply":
            result = image1 * (1 - opacity) + (image1 * image2) * opacity
        elif blend_mode == "screen":
            result = image1 * (1 - opacity) + (1 - (1 - image1) * (1 - image2)) * opacity
        elif blend_mode == "overlay":
            mask = image1 < 0.5
            overlay = torch.where(mask, 2 * image1 * image2, 1 - 2 * (1 - image1) * (1 - image2))
            result = image1 * (1 - opacity) + overlay * opacity
        elif blend_mode == "soft_light":
            mask = image2 < 0.5
            soft = torch.where(mask, 2 * image1 * image2 + image1**2 * (1 - 2 * image2), 
                             2 * image1 * (1 - image2) + torch.sqrt(image1) * (2 * image2 - 1))
            result = image1 * (1 - opacity) + soft * opacity
        elif blend_mode == "hard_light":
            mask = image2 < 0.5
            hard = torch.where(mask, 2 * image1 * image2, 1 - 2 * (1 - image1) * (1 - image2))
            result = image1 * (1 - opacity) + hard * opacity
        elif blend_mode == "difference":
            result = image1 * (1 - opacity) + torch.abs(image1 - image2) * opacity
        elif blend_mode == "add":
            result = torch.clamp(image1 + image2 * opacity, 0, 1)
        elif blend_mode == "subtract":
            result = torch.clamp(image1 - image2 * opacity, 0, 1)
        
        return (torch.clamp(result, 0, 1),)


class RKImageColorAdjust:
    """Adjust brightness, contrast, saturation, and hue of an image."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("adjusted_image",)
    CATEGORY = "rk_pix/image"
    FUNCTION = "adjust"
    
    def adjust(self, image, brightness=1.0, contrast=1.0, saturation=1.0, gamma=1.0):
        # Process each image in batch
        result = []
        for img in image:
            # Convert to PIL
            pil_img = Image.fromarray(
                np.clip(255.0 * img.cpu().numpy(), 0, 255).astype(np.uint8)
            )
            
            # Brightness
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(pil_img)
                pil_img = enhancer.enhance(brightness)
            
            # Contrast
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(pil_img)
                pil_img = enhancer.enhance(contrast)
            
            # Saturation
            if saturation != 1.0:
                enhancer = ImageEnhance.Color(pil_img)
                pil_img = enhancer.enhance(saturation)
            
            # Gamma
            if gamma != 1.0:
                arr = np.array(pil_img).astype(np.float32) / 255.0
                arr = np.clip(arr ** (1.0 / gamma), 0, 1)
                pil_img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
            
            # Back to tensor
            tensor = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)
            result.append(tensor)
        
        return (torch.stack(result),)


class RKImageTile:
    """Tile an image in a grid pattern."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "columns": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "rows": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "gap": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("tiled_image",)
    CATEGORY = "rk_pix/image"
    FUNCTION = "tile"
    
    def tile(self, image, columns, rows, gap):
        b, h, w, c = image.shape
        
        # Create output canvas
        out_h = h * rows + gap * (rows - 1)
        out_w = w * columns + gap * (columns - 1)
        output = torch.ones((b, out_h, out_w, c), dtype=image.dtype, device=image.device)
        
        for row in range(rows):
            for col in range(columns):
                y = row * (h + gap)
                x = col * (w + gap)
                output[:, y:y+h, x:x+w, :] = image
        
        return (output,)


class RKImageMirror:
    """Mirror/flip an image horizontally or vertically."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "direction": (["horizontal", "vertical", "both"], {"default": "horizontal"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mirrored_image",)
    CATEGORY = "rk_pix/image"
    FUNCTION = "mirror"
    
    def mirror(self, image, direction):
        if direction == "horizontal":
            return (torch.flip(image, dims=[2]),)
        elif direction == "vertical":
            return (torch.flip(image, dims=[1]),)
        elif direction == "both":
            return (torch.flip(image, dims=[1, 2]),)


class RKImageCrop:
    """Crop an image with percentage-based coordinates."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "right": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bottom": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_image",)
    CATEGORY = "rk_pix/image"
    FUNCTION = "crop"
    
    def crop(self, image, left, top, right, bottom):
        b, h, w, c = image.shape
        
        x1 = int(left * w)
        y1 = int(top * h)
        x2 = int(right * w)
        y2 = int(bottom * h)
        
        # Ensure valid coordinates
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(x1, min(x2, w))
        y2 = max(y1, min(y2, h))
        
        return (image[:, y1:y2, x1:x2, :],)


class RKImageBorder:
    """Add a border to an image."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "border_size": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "border_color": ("STRING", {"default": "#FFFFFF"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("bordered_image",)
    CATEGORY = "rk_pix/image"
    FUNCTION = "add_border"
    
    def add_border(self, image, border_size, border_color):
        b, h, w, c = image.shape
        
        # Parse color
        border_color = border_color.lstrip('#')
        r = int(border_color[0:2], 16) / 255.0
        g = int(border_color[2:4], 16) / 255.0
        bl = int(border_color[4:6], 16) / 255.0
        
        # Create output with border
        out_h = h + 2 * border_size
        out_w = w + 2 * border_size
        output = torch.ones((b, out_h, out_w, c), dtype=image.dtype, device=image.device)
        output[:, :, :, 0] = r
        output[:, :, :, 1] = g
        output[:, :, :, 2] = bl
        
        # Place original image in center
        output[:, border_size:border_size+h, border_size:border_size+w, :] = image
        
        return (output,)


# ============================================================
# RK ADVANCED IMAGE LOADER - Universal batch/frame-by-frame loader
# ============================================================

class RKAdvancedImageLoader:
    """
    Universal image loader with batch and frame-by-frame playback support.
    
    Features:
    - Load any image format (PNG, JPG, WEBP, BMP, TIFF, TGA, EXR, HDR, etc.)
    - Batch mode: load all images from a folder matching a pattern
    - Frame-by-frame mode: each run outputs the next frame (like a player)
    - EXR support with color space conversion (Linear, sRGB, custom gamma, custom LUT)
    - Auto-convert EXR to PNG with configurable bit depth
    - Start/end frame range control
    - Enable/disable toggle
    """

    _player_states = {}  # unique_id -> {current_index, start_index}

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        
        # Collect all image files from input directory
        all_files = []
        if os.path.exists(input_dir):
            for f in sorted(os.listdir(input_dir)):
                f_lower = f.lower()
                if any(f_lower.endswith(ext) for ext in [
                    '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif',
                    '.tga', '.dds', '.exr', '.hdr', '.pic', '.pnm', '.ppm',
                    '.pgm', '.pbm', '.pfm', '.sgi', '.ras', '.sun', '.ico',
                    '.icns', '.psd', '.xpm', '.xbm'
                ]):
                    all_files.append(f)
        
        return {
            "required": {
                "image": (sorted(all_files), {"tooltip": "Select an image file. For batch mode, select any file in the folder as reference."}),
                "enable": ("BOOLEAN", {"default": True, "tooltip": "Enable/disable this node"}),
                "mode": (["single", "batch_folder", "batch_pattern", "frame_by_frame"], {"default": "single", "tooltip": "single=load one image, batch_folder=load all images in same folder, batch_pattern=load matching pattern, frame_by_frame=play one frame per run"}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1, "tooltip": "Starting frame index (0=first)"}),
                "end_frame": ("INT", {"default": -1, "min": -1, "max": 99999, "step": 1, "tooltip": "End frame index (-1=last frame)"}),
                "step": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "Frame step size"}),
                "loop": ("BOOLEAN", {"default": True, "tooltip": "Loop back to start after reaching end"}),
                "reset": ("BOOLEAN", {"default": False, "tooltip": "Reset to start frame on next run"}),
            },
            "optional": {
                "pattern": ("STRING", {"default": "*.png", "tooltip": "Glob pattern for batch_pattern mode (e.g., *.png, frame_*.jpg, render_*.exr)"}),
                "sort_by": (["name", "name_natural", "modified_time", "size"], {"default": "name_natural", "tooltip": "How to sort batch files"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "current_frame", "total_frames", "info", "filepath")
    CATEGORY = "rk_pix/image"
    FUNCTION = "load_image"

    def load_image(self, image, enable, mode, start_frame, end_frame, step, loop, reset,
                   pattern="*.png", sort_by="name_natural", unique_id=None):
        
        if not enable:
            # Return a single black pixel image when disabled
            dummy = torch.zeros((1, 64, 64, 3))
            mask = torch.zeros((1, 64, 64))
            return (dummy, mask, 0, 0, "[DISABLED] Node is off", "")

        input_dir = folder_paths.get_input_directory()
        selected_path = os.path.join(input_dir, image)
        
        if not os.path.exists(selected_path):
            raise Exception(f"[RKAdvancedImageLoader] File not found: {selected_path}")

        # Determine file list based on mode
        if mode == "single":
            file_list = [selected_path]
        elif mode == "batch_folder":
            folder = os.path.dirname(selected_path)
            # Get all supported image files in the same folder
            all_exts = [
                '*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp', '*.tiff', '*.tif',
                '*.tga', '*.dds', '*.exr', '*.hdr', '*.pic', '*.pnm', '*.ppm',
                '*.pgm', '*.pbm', '*.pfm', '*.sgi', '*.ras', '*.sun', '*.ico',
                '*.psd', '*.xpm', '*.xbm'
            ]
            file_list = []
            for ext in all_exts:
                file_list.extend(glob.glob(os.path.join(folder, ext)))
                file_list.extend(glob.glob(os.path.join(folder, ext.upper())))
            # Remove duplicates and sort
            file_list = sorted(list(set(file_list)))
        elif mode in ("batch_pattern", "frame_by_frame"):
            folder = os.path.dirname(selected_path)
            if os.path.isabs(pattern):
                search_path = pattern
            else:
                search_path = os.path.join(folder, pattern)
            file_list = glob.glob(search_path)
            file_list = sorted(list(set(file_list)))
        else:
            file_list = [selected_path]

        if not file_list:
            raise Exception(f"[RKAdvancedImageLoader] No files found for mode '{mode}' with pattern '{pattern}'")

        # Sort files
        if sort_by == "name":
            file_list.sort()
        elif sort_by == "name_natural":
            file_list.sort(key=self._natural_sort_key)
        elif sort_by == "modified_time":
            file_list.sort(key=lambda x: os.path.getmtime(x))
        elif sort_by == "size":
            file_list.sort(key=lambda x: os.path.getsize(x))

        total_files = len(file_list)

        # Apply start/end range
        effective_start = min(start_frame, total_files - 1)
        effective_end = end_frame if end_frame >= 0 else total_files - 1
        effective_end = min(effective_end, total_files - 1)

        if effective_start > effective_end:
            raise Exception(f"[RKAdvancedImageLoader] Start frame {effective_start} > end frame {effective_end}")

        # Slice the file list
        ranged_files = file_list[effective_start:effective_end + 1:step]
        ranged_total = len(ranged_files)

        if mode == "frame_by_frame":
            # Frame-by-frame playback logic
            state_key = unique_id if unique_id else "default"
            
            if reset:
                self._player_states[state_key] = {"current": 0, "start": 0}
                current_idx = 0
            else:
                if state_key not in self._player_states:
                    self._player_states[state_key] = {"current": 0, "start": 0}
                
                state = self._player_states[state_key]
                current_idx = state["current"] % ranged_total if ranged_total > 0 else 0
            
            current_file = ranged_files[current_idx]
            img_tensor, mask_tensor, info_text = self._load_single_file(current_file)
            
            # Calculate next index
            next_idx = current_idx + 1
            if next_idx >= ranged_total:
                if loop:
                    next_idx = 0
                else:
                    next_idx = ranged_total - 1
            
            self._player_states[state_key]["current"] = next_idx
            
            actual_frame_num = effective_start + (current_idx * step)
            info = f"▶ Frame {current_idx + 1}/{ranged_total} (File #{actual_frame_num + 1}/{total_files}) | {info_text}"
            
            return (img_tensor, mask_tensor, current_idx, ranged_total, info, current_file)
        
        else:
            # Batch mode: load all images in the range
            images = []
            masks = []
            
            for fpath in ranged_files:
                img_tensor, mask_tensor, _ = self._load_single_file(fpath)
                images.append(img_tensor)
                masks.append(mask_tensor)
            
            # Stack into batch
            batch_images = torch.cat(images, dim=0)
            batch_masks = torch.cat(masks, dim=0)
            
            info = f"Batch: {ranged_total} files | Range: {effective_start}-{effective_end} | Step: {step} | Total in folder: {total_files}"
            
            return (batch_images, batch_masks, 0, ranged_total, info, file_list[0] if file_list else "")

    def _load_single_file(self, filepath):
        """Load a single image file and return (image_tensor, mask_tensor, info)."""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.exr':
            return self._load_exr(filepath)
        elif ext in ('.hdr', '.pic'):
            return self._load_hdr(filepath)
        else:
            return self._load_standard(filepath)

    def _load_standard(self, filepath):
        """Load standard image formats using PIL."""
        img = Image.open(filepath)
        
        # Handle animated formats by taking first frame
        if getattr(img, 'is_animated', False):
            img.seek(0)
        
        # Convert to RGB/RGBA
        if img.mode == 'RGBA':
            rgb = img.convert('RGB')
            alpha = np.array(img.split()[-1]).astype(np.float32) / 255.0
            mask = torch.from_numpy(alpha).unsqueeze(0)
            img_arr = np.array(rgb).astype(np.float32) / 255.0
        elif img.mode == 'L':
            img_arr = np.array(img.convert('RGB')).astype(np.float32) / 255.0
            mask = torch.ones((1, img_arr.shape[0], img_arr.shape[1]))
        elif img.mode == 'LA':
            rgb = img.convert('RGB')
            alpha = np.array(img.split()[-1]).astype(np.float32) / 255.0
            mask = torch.from_numpy(alpha).unsqueeze(0)
            img_arr = np.array(rgb).astype(np.float32) / 255.0
        elif img.mode == 'I':
            # 32-bit integer - normalize
            arr = np.array(img)
            img_arr = np.clip(arr.astype(np.float32) / 65535.0, 0, 1)
            img_arr = np.stack([img_arr] * 3, axis=-1) if len(img_arr.shape) == 2 else img_arr
            if img_arr.shape[-1] == 1:
                img_arr = np.repeat(img_arr, 3, axis=-1)
            mask = torch.ones((1, img_arr.shape[0], img_arr.shape[1]))
        elif img.mode == 'F':
            # 32-bit float
            arr = np.array(img)
            img_arr = np.clip(arr.astype(np.float32), 0, 1)
            img_arr = np.stack([img_arr] * 3, axis=-1) if len(img_arr.shape) == 2 else img_arr
            if img_arr.shape[-1] == 1:
                img_arr = np.repeat(img_arr, 3, axis=-1)
            mask = torch.ones((1, img_arr.shape[0], img_arr.shape[1]))
        elif img.mode == 'CMYK':
            img_arr = np.array(img.convert('RGB')).astype(np.float32) / 255.0
            mask = torch.ones((1, img_arr.shape[0], img_arr.shape[1]))
        else:
            img = img.convert('RGB')
            img_arr = np.array(img).astype(np.float32) / 255.0
            mask = torch.ones((1, img_arr.shape[0], img_arr.shape[1]))
        
        # Ensure shape is [H, W, 3]
        if len(img_arr.shape) == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)
        elif img_arr.shape[-1] == 4:
            img_arr = img_arr[:, :, :3]
        elif img_arr.shape[-1] == 1:
            img_arr = np.repeat(img_arr, 3, axis=-1)
        
        tensor = torch.from_numpy(img_arr).unsqueeze(0)
        
        info = f"{os.path.basename(filepath)} | {tensor.shape[2]}x{tensor.shape[1]} | {os.path.splitext(filepath)[1].upper()}"
        return (tensor, mask, info)

    def _load_exr(self, filepath):
        """Load EXR file with color space handling."""
        if OPENEXR_AVAILABLE:
            return self._load_exr_openexr(filepath)
        elif CV2_AVAILABLE:
            return self._load_exr_cv2(filepath)
        else:
            raise Exception("[RKAdvancedImageLoader] No EXR loader available. Install OpenEXR or OpenCV with EXR support.")

    def _load_exr_openexr(self, filepath):
        """Load EXR using OpenEXR library (best quality)."""
        exr_file = OpenEXR.InputFile(filepath)
        header = exr_file.header()
        
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # Determine channel format
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        
        # Read channels
        channels = header['channels'].keys()
        
        # Try to get RGB or RGBA
        r_str = exr_file.channel('R', pt) if 'R' in channels else None
        g_str = exr_file.channel('G', pt) if 'G' in channels else None
        b_str = exr_file.channel('B', pt) if 'B' in channels else None
        a_str = exr_file.channel('A', pt) if 'A' in channels else None
        
        # Fallback to Y if no RGB
        if r_str is None and 'Y' in channels:
            y_str = exr_file.channel('Y', pt)
            y = np.frombuffer(y_str, dtype=np.float32).reshape((height, width))
            r_str = y.tobytes()
            g_str = y.tobytes()
            b_str = y.tobytes()
        
        if r_str is None:
            exr_file.close()
            raise Exception(f"[RKAdvancedImageLoader] No recognizable channels in EXR: {channels}")
        
        r = np.frombuffer(r_str, dtype=np.float32).reshape((height, width))
        g = np.frombuffer(g_str, dtype=np.float32).reshape((height, width)) if g_str is not None else r.copy()
        b = np.frombuffer(b_str, dtype=np.float32).reshape((height, width)) if b_str is not None else r.copy()
        
        # Stack to RGB
        img_arr = np.stack([r, g, b], axis=-1)
        
        # Handle alpha
        if a_str is not None:
            a = np.frombuffer(a_str, dtype=np.float32).reshape((height, width))
            mask = torch.from_numpy(np.clip(a, 0, 1)).unsqueeze(0)
        else:
            mask = torch.ones((1, height, width))
        
        exr_file.close()
        
        # EXR is typically linear, may contain values > 1
        # Apply simple tone mapping / normalization for display
        max_val = img_arr.max()
        if max_val > 1.0:
            # ACES-inspired simple tone mapping
            img_arr = img_arr / (1.0 + img_arr)
        
        img_arr = np.clip(img_arr, 0, 1)
        tensor = torch.from_numpy(img_arr.astype(np.float32)).unsqueeze(0)
        
        info = f"{os.path.basename(filepath)} | {width}x{height} | EXR (OpenEXR) | Max: {max_val:.4f}"
        return (tensor, mask, info)

    def _load_exr_cv2(self, filepath):
        """Load EXR using OpenCV (fallback)."""
        img = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        
        if img is None:
            raise Exception(f"[RKAdvancedImageLoader] cv2 failed to load EXR: {filepath}")
        
        # cv2 loads as BGR
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.dtype == np.float32:
                # Already float
                img_arr = img
            else:
                # Half float or other - convert
                img_arr = img.astype(np.float32)
                if img_arr.max() > 1.0:
                    img_arr = img_arr / 65535.0
        else:
            # Grayscale
            img_arr = img.astype(np.float32)
            if img_arr.max() > 1.0:
                img_arr = img_arr / 65535.0
            img_arr = np.stack([img_arr] * 3, axis=-1)
        
        # Tone map if needed
        max_val = img_arr.max()
        if max_val > 1.0:
            img_arr = img_arr / (1.0 + img_arr)
        
        img_arr = np.clip(img_arr, 0, 1)
        h, w = img_arr.shape[:2]
        tensor = torch.from_numpy(img_arr).unsqueeze(0)
        mask = torch.ones((1, h, w))
        
        info = f"{os.path.basename(filepath)} | {w}x{h} | EXR (cv2) | Max: {max_val:.4f}"
        return (tensor, mask, info)

    def _load_hdr(self, filepath):
        """Load HDR/Radiance format."""
        if CV2_AVAILABLE:
            img = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is not None:
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_arr = img.astype(np.float32)
                # Tone map
                max_val = img_arr.max()
                if max_val > 1.0:
                    img_arr = img_arr / (1.0 + img_arr)
                img_arr = np.clip(img_arr, 0, 1)
                h, w = img_arr.shape[:2]
                tensor = torch.from_numpy(img_arr).unsqueeze(0)
                mask = torch.ones((1, h, w))
                info = f"{os.path.basename(filepath)} | {w}x{h} | HDR | Max: {max_val:.4f}"
                return (tensor, mask, info)
        
        # Fallback: try PIL
        return self._load_standard(filepath)

    def _natural_sort_key(self, s):
        """Natural sort key for filenames like frame_001, frame_002, etc."""
        import re
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split(r'([0-9]+)', s)]

    @classmethod
    def IS_CHANGED(cls, image, enable, mode, start_frame, end_frame, step, loop, reset,
                   pattern="*.png", sort_by="name_natural", unique_id=None):
        if not enable or mode != "frame_by_frame":
            # For batch/single modes, check if files changed
            input_dir = folder_paths.get_input_directory()
            filepath = os.path.join(input_dir, image)
            if os.path.exists(filepath):
                m = hashlib.sha256()
                with open(filepath, 'rb') as f:
                    m.update(f.read())
                return m.digest().hex()
        return float("NaN")


# ============================================================
# RK EXR CONVERTER - Dedicated EXR processing node
# ============================================================

class RK_EXRConverter:
    """
    Dedicated EXR converter with full color space control.
    
    Features:
    - Load EXR images with proper float handling
    - Convert to PNG with configurable bit depth (8, 16, 32)
    - Color space transforms: Linear, sRGB, ACES, Custom Gamma
    - Custom LUT support (3D LUT .cube files)
    - Tone mapping options
    - Auto-save converted PNG to output folder
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        
        exr_files = []
        if os.path.exists(input_dir):
            for f in sorted(os.listdir(input_dir)):
                if f.lower().endswith('.exr'):
                    exr_files.append(f)
        
        # Check for LUT files
        lut_files = ["None"]
        lut_dir = os.path.join(input_dir, "luts")
        if os.path.exists(lut_dir):
            for f in sorted(os.listdir(lut_dir)):
                if f.lower().endswith('.cube'):
                    lut_files.append(f)
        
        return {
            "required": {
                "exr_image": (sorted(exr_files) if exr_files else [""], {"tooltip": "Select EXR file to convert"}),
                "enable": ("BOOLEAN", {"default": True}),
                "color_space": (["linear", "srgb", "rec709", "aces", "custom_gamma", "custom_lut"], {"default": "linear", "tooltip": "Output color space"}),
                "gamma": ("FLOAT", {"default": 2.2, "min": 0.1, "max": 5.0, "step": 0.01, "tooltip": "Custom gamma value (when color_space=custom_gamma)"}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Exposure adjustment in stops"}),
                "tone_map": (["none", "reinhard", "aces", "filmic", "agx"], {"default": "aces", "tooltip": "Tone mapping algorithm"}),
                "output_depth": (["8bit", "16bit", "32bit"], {"default": "16bit", "tooltip": "Output bit depth for PNG"}),
                "auto_save_png": ("BOOLEAN", {"default": False, "tooltip": "Automatically save converted PNG to output folder"}),
                "filename_prefix": ("STRING", {"default": "exr_converted", "tooltip": "Prefix for saved PNG filename"}),
            },
            "optional": {
                "custom_lut": (sorted(lut_files), {"tooltip": "3D LUT .cube file (place in input/luts/ folder)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("converted_image", "info", "saved_path")
    CATEGORY = "rk_pix/image"
    FUNCTION = "convert_exr"
    OUTPUT_NODE = True

    def convert_exr(self, exr_image, enable, color_space, gamma, exposure, tone_map,
                    output_depth, auto_save_png, filename_prefix, custom_lut="None"):
        
        if not enable:
            dummy = torch.zeros((1, 64, 64, 3))
            return (dummy, "[DISABLED]", "")

        if not exr_image:
            raise Exception("[RK_EXRConverter] No EXR file selected")

        input_dir = folder_paths.get_input_directory()
        filepath = os.path.join(input_dir, exr_image)
        
        if not os.path.exists(filepath):
            raise Exception(f"[RK_EXRConverter] File not found: {filepath}")

        # Load EXR as float32 HDR
        img_arr, max_val, loader_info = self._load_exr_raw(filepath)
        h, w = img_arr.shape[:2]
        
        # Apply exposure
        if exposure != 0:
            img_arr = img_arr * (2.0 ** exposure)
        
        # Apply tone mapping
        img_arr = self._apply_tone_map(img_arr, tone_map)
        
        # Apply color space transform
        img_arr = self._apply_color_space(img_arr, color_space, gamma, custom_lut, input_dir)
        
        # Clip to valid range
        img_arr = np.clip(img_arr, 0, 1)
        
        # Convert to output bit depth
        if output_depth == "8bit":
            out_arr = (img_arr * 255.0).astype(np.uint8)
            depth_info = "8-bit"
        elif output_depth == "16bit":
            out_arr = (img_arr * 65535.0).astype(np.uint16)
            depth_info = "16-bit"
        else:  # 32bit - save as float PNG
            out_arr = img_arr.astype(np.float32)
            depth_info = "32-bit float"
        
        # Create tensor
        tensor = torch.from_numpy(img_arr.astype(np.float32)).unsqueeze(0)
        
        # Auto-save if enabled
        saved_path = ""
        if auto_save_png:
            output_dir = folder_paths.get_output_directory()
            counter = 1
            while True:
                filename = f"{filename_prefix}_{counter:04d}.png"
                save_path = os.path.join(output_dir, filename)
                if not os.path.exists(save_path):
                    break
                counter += 1
            
            # Save with PIL
            if output_depth == "32bit":
                # For 32-bit, save as 16-bit since most viewers don't support 32-bit PNG well
                save_arr = (img_arr * 65535.0).astype(np.uint16)
                pil_img = Image.fromarray(save_arr, mode='RGB')
            else:
                if len(out_arr.shape) == 2:
                    pil_img = Image.fromarray(out_arr, mode='L')
                else:
                    pil_img = Image.fromarray(out_arr, mode='RGB')
            
            pil_img.save(save_path)
            saved_path = save_path
        
        info = f"{os.path.basename(filepath)} | {w}x{h} | {loader_info} | {tone_map} tone map | {color_space} | {depth_info}"
        if saved_path:
            info += f" | Saved: {os.path.basename(saved_path)}"
        
        return (tensor, info, saved_path)

    def _load_exr_raw(self, filepath):
        """Load EXR and return raw float32 array + max value info."""
        if OPENEXR_AVAILABLE:
            exr_file = OpenEXR.InputFile(filepath)
            header = exr_file.header()
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            channels = header['channels'].keys()
            
            r_str = exr_file.channel('R', pt) if 'R' in channels else None
            g_str = exr_file.channel('G', pt) if 'G' in channels else None
            b_str = exr_file.channel('B', pt) if 'B' in channels else None
            
            if r_str is None and 'Y' in channels:
                y_str = exr_file.channel('Y', pt)
                y = np.frombuffer(y_str, dtype=np.float32).reshape((height, width))
                r_str = y.tobytes()
                g_str = y.tobytes()
                b_str = y.tobytes()
            
            r = np.frombuffer(r_str, dtype=np.float32).reshape((height, width))
            g = np.frombuffer(g_str, dtype=np.float32).reshape((height, width)) if g_str is not None else r.copy()
            b = np.frombuffer(b_str, dtype=np.float32).reshape((height, width)) if b_str is not None else r.copy()
            
            img_arr = np.stack([r, g, b], axis=-1)
            exr_file.close()
            max_val = img_arr.max()
            return img_arr, max_val, f"OpenEXR | Max: {max_val:.4f}"
        
        elif CV2_AVAILABLE:
            img = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                raise Exception(f"cv2 failed to load EXR: {filepath}")
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.stack([img] * 3, axis=-1)
            img_arr = img.astype(np.float32)
            max_val = img_arr.max()
            return img_arr, max_val, f"cv2 | Max: {max_val:.4f}"
        else:
            raise Exception("No EXR loader available")

    def _apply_tone_map(self, img_arr, tone_map):
        """Apply tone mapping to HDR image."""
        if tone_map == "none":
            return img_arr
        elif tone_map == "reinhard":
            return img_arr / (1.0 + img_arr)
        elif tone_map == "aces":
            # Simplified ACES tone mapping
            a = 2.51
            b = 0.03
            c = 2.43
            d = 0.59
            e = 0.14
            return np.clip((img_arr * (a * img_arr + b)) / (img_arr * (c * img_arr + d) + e), 0, 1)
        elif tone_map == "filmic":
            # Filmic tone mapping
            img_arr = np.maximum(img_arr, 0.0)
            x = np.maximum(img_arr, 0.004)
            return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)
        elif tone_map == "agx":
            # AgX-like approximation
            img_arr = np.maximum(img_arr, 0.0)
            return img_arr / (img_arr + 0.5)
        return img_arr

    def _apply_color_space(self, img_arr, color_space, gamma, custom_lut, input_dir):
        """Apply color space transform."""
        if color_space == "linear":
            return img_arr
        elif color_space == "srgb":
            # Linear to sRGB
            mask = img_arr <= 0.0031308
            result = np.where(mask, img_arr * 12.92, 1.055 * (img_arr ** (1.0 / 2.4)) - 0.055)
            return result
        elif color_space == "rec709":
            mask = img_arr <= 0.018
            return np.where(mask, img_arr * 4.5, 1.099 * (img_arr ** 0.45) - 0.099)
        elif color_space == "aces":
            # Simplified ACES sRGB output transform
            img_arr = np.maximum(img_arr, 0.0)
            a = 2.51
            b = 0.03
            c = 2.43
            d = 0.59
            e = 0.14
            return np.clip((img_arr * (a * img_arr + b)) / (img_arr * (c * img_arr + d) + e), 0, 1)
        elif color_space == "custom_gamma":
            return img_arr ** (1.0 / gamma)
        elif color_space == "custom_lut":
            if custom_lut != "None":
                lut_path = os.path.join(input_dir, "luts", custom_lut)
                if os.path.exists(lut_path):
                    return self._apply_3d_lut(img_arr, lut_path)
            # Fallback to sRGB if LUT not found
            mask = img_arr <= 0.0031308
            return np.where(mask, img_arr * 12.92, 1.055 * (img_arr ** (1.0 / 2.4)) - 0.055)
        return img_arr

    def _apply_3d_lut(self, img_arr, lut_path):
        """Apply a 3D LUT (.cube file) to the image."""
        try:
            lut_size, lut_data = self._parse_cube_lut(lut_path)
            
            # Reshape image for LUT application
            h, w, c = img_arr.shape
            pixels = img_arr.reshape(-1, 3)
            
            # Clamp to LUT range (typically 0-1)
            pixels = np.clip(pixels, 0, 1)
            
            # Scale to LUT indices
            scale = lut_size - 1
            
            # Apply trilinear interpolation
            result = np.zeros_like(pixels)
            for i in range(len(pixels)):
                r, g, b = pixels[i]
                
                # Get lower and upper indices
                r0, g0, b0 = int(r * scale), int(g * scale), int(b * scale)
                r1, g1, b1 = min(r0 + 1, lut_size - 1), min(g0 + 1, lut_size - 1), min(b0 + 1, lut_size - 1)
                
                # Fractional parts
                fr, fg, fb = (r * scale) - r0, (g * scale) - g0, (b * scale) - b0
                
                # 8 corner samples
                c000 = lut_data[b0, g0, r0]
                c001 = lut_data[b1, g0, r0]
                c010 = lut_data[b0, g1, r0]
                c011 = lut_data[b1, g1, r0]
                c100 = lut_data[b0, g0, r1]
                c101 = lut_data[b1, g0, r1]
                c110 = lut_data[b0, g1, r1]
                c111 = lut_data[b1, g1, r1]
                
                # Trilinear interpolation
                result[i] = (
                    c000 * (1 - fr) * (1 - fg) * (1 - fb) +
                    c001 * (1 - fr) * (1 - fg) * fb +
                    c010 * (1 - fr) * fg * (1 - fb) +
                    c011 * (1 - fr) * fg * fb +
                    c100 * fr * (1 - fg) * (1 - fb) +
                    c101 * fr * (1 - fg) * fb +
                    c110 * fr * fg * (1 - fb) +
                    c111 * fr * fg * fb
                )
            
            return result.reshape(h, w, 3)
        except Exception as e:
            print(f"[RK_EXRConverter] LUT application failed: {e}, falling back to linear")
            return img_arr

    def _parse_cube_lut(self, lut_path):
        """Parse a .cube 3D LUT file."""
        lut_size = 0
        lut_data = []
        
        with open(lut_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('LUT_3D_SIZE'):
                    lut_size = int(line.split()[-1])
                elif line.startswith('LUT_1D_SIZE'):
                    lut_size = int(line.split()[-1])
                elif line.startswith('DOMAIN_'):
                    continue
                elif line.startswith('TITLE'):
                    continue
                else:
                    # RGB values
                    vals = [float(v) for v in line.split()]
                    if len(vals) >= 3:
                        lut_data.append(vals[:3])
        
        if lut_size == 0:
            lut_size = int(round(len(lut_data) ** (1/3)))
        
        lut_array = np.array(lut_data, dtype=np.float32)
        lut_array = lut_array.reshape((lut_size, lut_size, lut_size, 3))
        
        return lut_size, lut_array

    @classmethod
    def IS_CHANGED(cls, exr_image, enable, color_space, gamma, exposure, tone_map,
                   output_depth, auto_save_png, filename_prefix, custom_lut="None"):
        if not enable:
            return ""
        input_dir = folder_paths.get_input_directory()
        filepath = os.path.join(input_dir, exr_image)
        if os.path.exists(filepath):
            m = hashlib.sha256()
            with open(filepath, 'rb') as f:
                m.update(f.read())
            return m.digest().hex()
        return float("NaN")


# ============================================================
# REGISTRATION
# ============================================================

IMAGE_CLASS_MAPPINGS = {
    "RKImageBlend": RKImageBlend,
    "RKImageColorAdjust": RKImageColorAdjust,
    "RKImageTile": RKImageTile,
    "RKImageMirror": RKImageMirror,
    "RKImageCrop": RKImageCrop,
    "RKImageBorder": RKImageBorder,
    "RKAdvancedImageLoader": RKAdvancedImageLoader,
    "RK_EXRConverter": RK_EXRConverter,
}

IMAGE_NAME_MAPPINGS = {
    "RKImageBlend": "Blend Images (RK)",
    "RKImageColorAdjust": "Color Adjust (RK)",
    "RKImageTile": "Tile Image (RK)",
    "RKImageMirror": "Mirror Image (RK)",
    "RKImageCrop": "Crop Image (RK)",
    "RKImageBorder": "Add Border (RK)",
    "RKAdvancedImageLoader": "Advanced Image Loader (RK)",
    "RK_EXRConverter": "EXR Converter (RK)",
}
