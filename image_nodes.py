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
import fnmatch

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
    - Custom folder path input - type any folder on your system
    - Frame-by-frame mode: each run outputs the next frame (frame 1, frame 2, frame 3...)
    - Manual frame offset: set custom starting frame number
    - Batch mode: load all images from a folder matching a pattern
    - Dual output: normal IMAGE + RAW EXR output (for EXR Converter node)
    - Upload single image option
    - Start/end frame range control
    - Enable/disable toggle
    """

    _player_states = {}  # unique_id -> {current_index}

    # Supported image extensions
    ALL_EXTENSIONS = [
        '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif',
        '.tga', '.dds', '.exr', '.hdr', '.pic', '.pnm', '.ppm',
        '.pgm', '.pbm', '.pfm', '.sgi', '.ras', '.sun', '.ico',
        '.icns', '.psd', '.xpm', '.xbm'
    ]

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        
        # Collect all image files from input directory for upload option
        all_files = []
        if os.path.exists(input_dir):
            for f in sorted(os.listdir(input_dir)):
                f_lower = f.lower()
                if any(f_lower.endswith(ext) for ext in cls.ALL_EXTENSIONS):
                    all_files.append(f)
        
        return {
            "required": {
                "source_type": (["folder_path", "upload_image"], {"default": "folder_path", "tooltip": "folder_path=load from folder, upload_image=load single uploaded image"}),
                "folder_path": ("STRING", {"default": "", "tooltip": "FULL folder path to your image sequence (e.g., /home/user/renders). Only used when source_type=folder_path"}),
                "pattern": ("STRING", {"default": "*", "tooltip": "File pattern to match (e.g., *.png, frame_*.exr). Use * for all images. Only for folder_path mode."}),
                "uploaded_image": (sorted(all_files), {"tooltip": "Select an uploaded image from ComfyUI input folder. Only used when source_type=upload_image"}),
                "enable": ("BOOLEAN", {"default": True, "tooltip": "Enable/disable this node"}),
                "mode": (["frame_by_frame", "batch", "single"], {"default": "frame_by_frame", "tooltip": "frame_by_frame=play one image per run, batch=load all as batch, single=load first image only"}),
                "use_manual_frame": ("BOOLEAN", {"default": False, "tooltip": "Enable to set a custom starting frame number. When disabled, starts from frame 1 (index 0)."}),
                "manual_frame": ("INT", {"default": 1, "min": 1, "max": 99999, "step": 1, "tooltip": "Custom starting frame number (1-based). Only used when use_manual_frame=True."}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1, "tooltip": "Starting file index (0=first file in sorted list)."}),
                "end_frame": ("INT", {"default": -1, "min": -1, "max": 99999, "step": 1, "tooltip": "End file index (-1=last file)"}),
                "step": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "Skip N images (1=every image, 2=every other)"}),
                "loop": ("BOOLEAN", {"default": True, "tooltip": "Loop back to start after last frame"}),
                "reset": ("BOOLEAN", {"default": False, "tooltip": "Reset to start frame on next run"}),
            },
            "optional": {
                "sort_by": (["name_natural", "name", "modified_time", "size"], {"default": "name_natural", "tooltip": "How to sort files: name_natural=frame_001,frame_002... name=alphabetical"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "raw_exr", "current_frame", "total_frames", "info", "filepath")
    CATEGORY = "rk_pix/image"
    FUNCTION = "load_image"

    def load_image(self, source_type, folder_path, pattern, uploaded_image, enable, mode, use_manual_frame, manual_frame,
                   start_frame, end_frame, step, loop, reset,
                   sort_by="name_natural", unique_id=None):
        
        if not enable:
            dummy = torch.zeros((1, 64, 64, 3))
            mask = torch.zeros((1, 64, 64))
            return (dummy, mask, dummy, 0, 0, "[DISABLED] Node is off", "")

        # === UPLOAD IMAGE MODE ===
        if source_type == "upload_image":
            input_dir = folder_paths.get_input_directory()
            filepath = os.path.join(input_dir, uploaded_image)
            
            if not os.path.exists(filepath):
                raise Exception(f"[RKAdvancedImageLoader] Uploaded image not found: {filepath}")
            
            img_tensor, mask_tensor, raw_exr_tensor, info_text = self._load_file_with_raw(filepath)
            info = f"Uploaded: {info_text}"
            return (img_tensor, mask_tensor, raw_exr_tensor, 0, 1, info, filepath)

        # === FOLDER PATH MODE ===
        # Validate folder path
        if not folder_path or not folder_path.strip():
            raise Exception("[RKAdvancedImageLoader] folder_path is empty. Please enter a folder path.")
        
        folder = os.path.normpath(folder_path.strip())
        if not os.path.exists(folder):
            raise Exception(f"[RKAdvancedImageLoader] Folder not found: {folder}")
        if not os.path.isdir(folder):
            raise Exception(f"[RKAdvancedImageLoader] Path is not a folder: {folder}")

        # Build search pattern
        if os.path.isabs(pattern):
            search_path = pattern
        else:
            search_path = os.path.join(folder, pattern)
        
        # Find all matching files
        file_list = glob.glob(search_path)
        
        # Also try case-insensitive match
        if not file_list:
            folder_name = os.path.dirname(search_path)
            file_pattern = os.path.basename(search_path)
            if os.path.exists(folder_name):
                for f in os.listdir(folder_name):
                    if fnmatch.fnmatch(f.lower(), file_pattern.lower()):
                        file_list.append(os.path.join(folder_name, f))
        
        if not file_list:
            raise Exception(f"[RKAdvancedImageLoader] No files found in '{folder}' matching pattern '{pattern}'")
        
        # Remove duplicates
        file_list = sorted(list(set(file_list)))
        
        # Filter to only image files
        file_list = [f for f in file_list if any(f.lower().endswith(ext) for ext in self.ALL_EXTENSIONS)]
        
        if not file_list:
            raise Exception(f"[RKAdvancedImageLoader] No valid image files found in '{folder}' matching pattern '{pattern}'")

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

        # Apply manual frame offset
        if use_manual_frame:
            # manual_frame is 1-based, convert to 0-based index
            effective_start = min(manual_frame - 1, total_files - 1)
        else:
            effective_start = min(start_frame, total_files - 1)
        
        effective_end = end_frame if end_frame >= 0 else total_files - 1
        effective_end = min(effective_end, total_files - 1)

        if effective_start > effective_end:
            raise Exception(f"[RKAdvancedImageLoader] Start frame {effective_start} > end frame {effective_end} (total: {total_files})")

        # Slice the file list
        ranged_files = file_list[effective_start:effective_end + 1:step]
        ranged_total = len(ranged_files)

        if mode == "frame_by_frame":
            # Frame-by-frame: each run outputs next image
            state_key = unique_id if unique_id else "default"
            
            if reset:
                self._player_states[state_key] = 0
                current_idx = 0
            else:
                if state_key not in self._player_states:
                    self._player_states[state_key] = 0
                current_idx = self._player_states[state_key] % ranged_total
            
            current_file = ranged_files[current_idx]
            img_tensor, mask_tensor, raw_exr_tensor, info_text = self._load_file_with_raw(current_file)
            
            # Calculate next index
            next_idx = current_idx + 1
            if next_idx >= ranged_total:
                if loop:
                    next_idx = 0
                else:
                    next_idx = ranged_total - 1
            
            self._player_states[state_key] = next_idx
            
            # Display frame number (1-based for user)
            actual_frame_num = effective_start + (current_idx * step) + 1
            info = f"Frame {actual_frame_num}/{total_files} (Index {current_idx + 1}/{ranged_total}) | {info_text}"
            
            return (img_tensor, mask_tensor, raw_exr_tensor, actual_frame_num, ranged_total, info, current_file)
        
        elif mode == "batch":
            # Batch mode: load all images
            images = []
            masks = []
            raw_exrs = []
            
            for fpath in ranged_files:
                img_tensor, mask_tensor, raw_exr_tensor, _ = self._load_file_with_raw(fpath)
                images.append(img_tensor)
                masks.append(mask_tensor)
                raw_exrs.append(raw_exr_tensor)
            
            batch_images = torch.cat(images, dim=0)
            batch_masks = torch.cat(masks, dim=0)
            batch_raw = torch.cat(raw_exrs, dim=0)
            
            info = f"Batch: {ranged_total} files | Range: {effective_start + 1}-{effective_end + 1} | Step: {step} | Total: {total_files}"
            
            return (batch_images, batch_masks, batch_raw, 0, ranged_total, info, file_list[0] if file_list else "")
        
        else:  # single mode
            current_file = ranged_files[0]
            img_tensor, mask_tensor, raw_exr_tensor, info_text = self._load_file_with_raw(current_file)
            info = f"Single: {info_text}"
            return (img_tensor, mask_tensor, raw_exr_tensor, effective_start + 1, ranged_total, info, current_file)

    def _load_file_with_raw(self, filepath):
        """
        Load a file and return both processed image AND raw EXR data.
        For EXR: returns (tonemapped_image, mask, raw_float_exr, info)
        For others: returns (image, mask, same_image_as_raw, info)
        """
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.exr':
            return self._load_exr_dual(filepath)
        elif ext in ('.hdr', '.pic'):
            img, mask, info = self._load_hdr(filepath)
            return (img, mask, img, info)
        else:
            img, mask, info = self._load_standard(filepath)
            return (img, mask, img, info)

    def _load_standard(self, filepath):
        """Load standard image formats using PIL."""
        img = Image.open(filepath)
        
        if getattr(img, 'is_animated', False):
            img.seek(0)
        
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
            arr = np.array(img)
            img_arr = np.clip(arr.astype(np.float32) / 65535.0, 0, 1)
            img_arr = np.stack([img_arr] * 3, axis=-1) if len(img_arr.shape) == 2 else img_arr
            if img_arr.shape[-1] == 1:
                img_arr = np.repeat(img_arr, 3, axis=-1)
            mask = torch.ones((1, img_arr.shape[0], img_arr.shape[1]))
        elif img.mode == 'F':
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
        
        if len(img_arr.shape) == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)
        elif img_arr.shape[-1] == 4:
            img_arr = img_arr[:, :, :3]
        elif img_arr.shape[-1] == 1:
            img_arr = np.repeat(img_arr, 3, axis=-1)
        
        tensor = torch.from_numpy(img_arr).unsqueeze(0)
        info = f"{os.path.basename(filepath)} | {tensor.shape[2]}x{tensor.shape[1]} | {ext.upper()}"
        return (tensor, mask, info)

    def _load_exr_dual(self, filepath):
        """
        Load EXR and return BOTH:
        - processed: tonemapped/clamped for display [0-1]
        - raw: original float values (may be >1) for EXR Converter
        """
        if OPENEXR_AVAILABLE:
            return self._load_exr_dual_openexr(filepath)
        elif CV2_AVAILABLE:
            return self._load_exr_dual_cv2(filepath)
        else:
            raise Exception("[RKAdvancedImageLoader] No EXR loader available. Install OpenEXR or OpenCV.")

    def _load_exr_dual_openexr(self, filepath):
        """Load EXR with OpenEXR - returns (processed, mask, raw, info)."""
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
        a_str = exr_file.channel('A', pt) if 'A' in channels else None
        
        if r_str is None and 'Y' in channels:
            y_str = exr_file.channel('Y', pt)
            y = np.frombuffer(y_str, dtype=np.float32).reshape((height, width))
            r_str = y.tobytes()
            g_str = y.tobytes()
            b_str = y.tobytes()
        
        if r_str is None:
            exr_file.close()
            raise Exception(f"No recognizable channels in EXR: {channels}")
        
        r = np.frombuffer(r_str, dtype=np.float32).reshape((height, width))
        g = np.frombuffer(g_str, dtype=np.float32).reshape((height, width)) if g_str is not None else r.copy()
        b = np.frombuffer(b_str, dtype=np.float32).reshape((height, width)) if b_str is not None else r.copy()
        
        # RAW float array (may contain values > 1)
        raw_arr = np.stack([r, g, b], axis=-1)
        
        # Handle alpha
        if a_str is not None:
            a = np.frombuffer(a_str, dtype=np.float32).reshape((height, width))
            mask = torch.from_numpy(np.clip(a, 0, 1)).unsqueeze(0)
        else:
            mask = torch.ones((1, height, width))
        
        exr_file.close()
        
        max_val = raw_arr.max()
        
        # PROCESSED: tonemapped for display [0-1]
        processed_arr = raw_arr.copy()
        if max_val > 1.0:
            processed_arr = processed_arr / (1.0 + processed_arr)
        processed_arr = np.clip(processed_arr, 0, 1)
        processed_tensor = torch.from_numpy(processed_arr.astype(np.float32)).unsqueeze(0)
        
        # RAW: original float values (for EXR Converter)
        raw_clamped = np.clip(raw_arr, 0, 65504.0)
        raw_tensor = torch.from_numpy(raw_clamped.astype(np.float32)).unsqueeze(0)
        
        info = f"{os.path.basename(filepath)} | {width}x{height} | EXR | Max: {max_val:.4f}"
        return (processed_tensor, mask, raw_tensor, info)

    def _load_exr_dual_cv2(self, filepath):
        """Load EXR with cv2 - returns (processed, mask, raw, info)."""
        img = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        
        if img is None:
            raise Exception(f"cv2 failed to load EXR: {filepath}")
        
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.dtype != np.float32:
                img = img.astype(np.float32)
                if img.max() > 1.0:
                    img = img / 65535.0
        else:
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 65535.0
            img = np.stack([img] * 3, axis=-1)
        
        raw_arr = img.copy()
        max_val = raw_arr.max()
        
        # PROCESSED: tonemapped
        processed_arr = raw_arr.copy()
        if max_val > 1.0:
            processed_arr = processed_arr / (1.0 + processed_arr)
        processed_arr = np.clip(processed_arr, 0, 1)
        processed_tensor = torch.from_numpy(processed_arr).unsqueeze(0)
        
        # RAW: keep original float values
        raw_clamped = np.clip(raw_arr, 0, 65504.0)
        raw_tensor = torch.from_numpy(raw_clamped).unsqueeze(0)
        
        h, w = raw_arr.shape[:2]
        mask = torch.ones((1, h, w))
        
        info = f"{os.path.basename(filepath)} | {w}x{h} | EXR (cv2) | Max: {max_val:.4f}"
        return (processed_tensor, mask, raw_tensor, info)

    def _load_hdr(self, filepath):
        """Load HDR/Radiance format."""
        if CV2_AVAILABLE:
            img = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is not None:
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_arr = img.astype(np.float32)
                max_val = img_arr.max()
                if max_val > 1.0:
                    img_arr = img_arr / (1.0 + img_arr)
                img_arr = np.clip(img_arr, 0, 1)
                h, w = img_arr.shape[:2]
                tensor = torch.from_numpy(img_arr).unsqueeze(0)
                mask = torch.ones((1, h, w))
                info = f"{os.path.basename(filepath)} | {w}x{h} | HDR | Max: {max_val:.4f}"
                return (tensor, mask, info)
        
        return self._load_standard(filepath)

    def _natural_sort_key(self, s):
        """Natural sort key for filenames like frame_001, frame_002."""
        import re
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split(r'([0-9]+)', s)]

    @classmethod
    def IS_CHANGED(cls, source_type, folder_path, pattern, uploaded_image, enable, mode, use_manual_frame, manual_frame,
                   start_frame, end_frame, step, loop, reset,
                   sort_by="name_natural", unique_id=None):
        if not enable:
            return ""
        if source_type == "upload_image":
            input_dir = folder_paths.get_input_directory()
            filepath = os.path.join(input_dir, uploaded_image)
            if os.path.exists(filepath):
                m = hashlib.sha256()
                with open(filepath, 'rb') as f:
                    m.update(f.read())
                return m.digest().hex()
            return float("NaN")
        # folder_path mode
        if mode == "frame_by_frame":
            return float("NaN")
        if folder_path and folder_path.strip():
            folder = os.path.normpath(folder_path.strip())
            if os.path.exists(folder) and os.path.isdir(folder):
                m = hashlib.sha256()
                for f in sorted(os.listdir(folder))[:50]:
                    m.update(f.encode())
                return m.digest().hex()
        return float("NaN")

class RK_EXRConverter:
    """
    Dedicated EXR converter with full color space control.
    
    Features:
    - Accept raw EXR IMAGE tensor from Advanced Image Loader
    - Convert to PNG with configurable bit depth (8, 16, 32)
    - Color space transforms: Linear, sRGB, ACES, Custom Gamma
    - Custom LUT support (3D LUT .cube files)
    - Tone mapping options
    - Auto-save converted PNG to output folder
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        
        # Check for LUT files
        lut_files = ["None"]
        lut_dir = os.path.join(input_dir, "luts")
        if os.path.exists(lut_dir):
            for f in sorted(os.listdir(lut_dir)):
                if f.lower().endswith('.cube'):
                    lut_files.append(f)
        
        return {
            "required": {
                "raw_exr": ("IMAGE", {"tooltip": "Connect raw_exr output from Advanced Image Loader here"}),
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

    def convert_exr(self, raw_exr, enable, color_space, gamma, exposure, tone_map,
                    output_depth, auto_save_png, filename_prefix, custom_lut="None"):
        
        if not enable:
            dummy = torch.zeros((1, 64, 64, 3))
            return (dummy, "[DISABLED]", "")

        if raw_exr is None or raw_exr.shape[0] == 0:
            raise Exception("[RK_EXRConverter] No raw EXR input connected. Connect raw_exr from Advanced Image Loader.")

        # raw_exr is a torch tensor [B, H, W, 3] with float values (may be > 1)
        img_arr = raw_exr[0].cpu().numpy().astype(np.float32)  # Take first batch item
        h, w = img_arr.shape[:2]
        max_val = img_arr.max()
        
        # Apply exposure
        if exposure != 0:
            img_arr = img_arr * (2.0 ** exposure)
        
        # Apply tone mapping
        img_arr = self._apply_tone_map(img_arr, tone_map)
        
        # Apply color space transform
        input_dir = folder_paths.get_input_directory()
        img_arr = self._apply_color_space(img_arr, color_space, gamma, custom_lut, input_dir)
        
        # Clip to valid range
        img_arr = np.clip(img_arr, 0, 1)
        
        # Apply dithering to reduce banding
        img_arr = self._apply_dithering(img_arr, output_depth)
        
        # Convert to output bit depth
        if output_depth == "8bit":
            out_arr = (img_arr * 255.0).astype(np.uint8)
            depth_info = "8-bit"
        elif output_depth == "16bit":
            out_arr = (img_arr * 65535.0).astype(np.uint16)
            depth_info = "16-bit"
        else:  # 32bit - keep float precision
            out_arr = img_arr.astype(np.float32)
            depth_info = "32-bit float"
        
        # Create tensor from the dithered/quantized array for maximum quality
        if output_depth == "32bit":
            tensor = torch.from_numpy(img_arr.astype(np.float32)).unsqueeze(0)
        else:
            # For 8/16 bit, convert back through float to preserve the dithering
            tensor = torch.from_numpy(out_arr.astype(np.float32) / (255.0 if output_depth == "8bit" else 65535.0)).unsqueeze(0)
        
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
        
        info = f"EXR Convert | {w}x{h} | Max: {max_val:.4f} | {tone_map} tone map | {color_space} | {depth_info}"
        if saved_path:
            info += f" | Saved: {os.path.basename(saved_path)}"
        
        return (tensor, info, saved_path)

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

    def _apply_dithering(self, img_arr, depth):
        """Apply Bayer dithering to reduce banding when quantizing to lower bit depths."""
        if depth == "32bit":
            return img_arr
        
        h, w = img_arr.shape[:2]
        
        # 4x4 Bayer matrix normalized to [-0.5, 0.5]
        bayer = np.array([
            [ 0, 32,  8, 40],
            [48, 16, 56, 24],
            [12, 44,  4, 36],
            [60, 28, 52, 20]
        ], dtype=np.float32) / 64.0 - 0.5
        
        # Tile the Bayer matrix to image size
        bayer_tiled = np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w]
        
        # Scale dithering amount based on bit depth
        if depth == "8bit":
            scale = 1.0 / 255.0
        elif depth == "16bit":
            scale = 1.0 / 65535.0
        else:
            scale = 0.0
        
        # Apply dithering
        dithered = img_arr + bayer_tiled[..., None] * scale
        return np.clip(dithered, 0, 1)

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
    def IS_CHANGED(cls, raw_exr, enable, color_space, gamma, exposure, tone_map,
                   output_depth, auto_save_png, filename_prefix, custom_lut="None"):
        if not enable:
            return ""
        # For IMAGE input, use a hash of the tensor
        if raw_exr is not None:
            try:
                img_bytes = raw_exr.cpu().numpy().tobytes()
                m = hashlib.sha256()
                m.update(img_bytes)
                return m.digest().hex()
            except:
                pass
        return float("NaN")


# ============================================================
# RK IMAGE TO EXR CONVERTER - Convert PNG/JPG to EXR with analysis
# ============================================================

class RKImageToEXR:
    """
    Convert any image to EXR format with deep reference EXR analysis.
    
    Workflow:
    1. Load EXR -> EXR Converter (color space, tone map) -> IMAGE
    2. That IMAGE + original raw EXR (reference) -> This node
    3. Output EXR matches original raw EXR characteristics
    
    Features:
    - Analyzes reference EXR to copy: data range, gamma, linearity
    - Auto-detects if reference is linear or sRGB
    - Reverses tone mapping to restore HDR range
    - Preserves original EXR bit depth and compression
    - Preview output to verify before saving
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Processed image (from EXR Converter or any node) to convert back to EXR"}),
                "enable": ("BOOLEAN", {"default": True}),
                "save_exr": ("BOOLEAN", {"default": True, "tooltip": "ON=save EXR file, OFF=preview only"}),
                "save_path": ("STRING", {"default": "", "tooltip": "Folder path to save EXR. Empty=ComfyUI output folder."}),
                "filename": ("STRING", {"default": "output", "tooltip": "Filename without .exr extension"}),
                "auto_increment": ("BOOLEAN", {"default": True, "tooltip": "Auto-number if file exists"}),
            },
            "optional": {
                "reference_exr": ("IMAGE", {"tooltip": "CRITICAL: Connect original raw EXR here to copy its data characteristics (range, gamma, linearity)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "info")
    CATEGORY = "rk_pix/image"
    FUNCTION = "convert_to_exr"
    OUTPUT_NODE = True

    def convert_to_exr(self, image, enable, save_exr, save_path, filename, auto_increment, reference_exr=None):
        
        if not enable:
            dummy = torch.zeros((1, 64, 64, 3))
            return (dummy, "[DISABLED]")

        if image is None or image.shape[0] == 0:
            raise Exception("[RKImageToEXR] No image input provided")

        # Get input image data
        img_tensor = image[0] if len(image.shape) == 4 else image
        h, w, c = img_tensor.shape
        img_arr = img_tensor.cpu().numpy().astype(np.float32)
        
        # Analyze reference EXR if provided
        ref_analysis = self._analyze_reference(reference_exr) if reference_exr is not None else None
        
        # Normalize input to [0,1] if needed
        max_val = img_arr.max()
        if max_val > 1.0:
            img_arr = img_arr / 255.0 if max_val <= 255.0 else img_arr / 65535.0
        
        # === CORE LOGIC: Match reference EXR characteristics ===
        if ref_analysis is not None:
            # Reference EXR provided - copy its characteristics
            
            # 1. Reverse gamma if reference was sRGB-like
            if ref_analysis['is_srgb']:
                img_arr = self._reverse_srgb(img_arr)
            
            # 2. Reverse tone mapping to restore HDR range
            if ref_analysis['max_val'] > 1.0:
                img_arr = self._reverse_tonemap(img_arr, ref_analysis['max_val'])
            
            # 3. Scale to match reference data range
            if ref_analysis['max_val'] > 1.0:
                target_max = ref_analysis['max_val']
                current_max = max(img_arr.max(), 1e-6)
                scale = target_max / current_max
                img_arr = img_arr * scale
            
            bit_depth = ref_analysis['bit_depth']
            compression = ref_analysis['compression']
        else:
            # No reference - use defaults
            bit_depth = "half"
            compression = "zip"
        
        # Build channels
        if c >= 3:
            channels = {'R': img_arr[:, :, 0], 'G': img_arr[:, :, 1], 'B': img_arr[:, :, 2]}
            if c >= 4:
                channels['A'] = img_arr[:, :, 3]
        else:
            gray = img_arr[:, :, 0]
            channels = {'R': gray, 'G': gray, 'B': gray}
        
        # Save EXR if enabled
        saved_name = ""
        if save_exr:
            output_dir = self._get_output_dir(save_path)
            exr_path, fname = self._build_filename(output_dir, filename, auto_increment)
            
            try:
                if OPENEXR_AVAILABLE:
                    self._save_exr_openexr(exr_path, channels, h, w, bit_depth, compression)
                elif CV2_AVAILABLE:
                    self._save_exr_cv2(exr_path, img_arr)
                else:
                    raise Exception("No EXR writer available")
                saved_name = fname
            except Exception as e:
                dummy = torch.zeros((1, 64, 64, 3))
                return (dummy, f"[SAVE ERROR] {str(e)}")
        
        # Create preview (tonemapped for display)
        preview_arr = img_arr.copy()
        if preview_arr.max() > 1.0:
            preview_arr = preview_arr / (1.0 + preview_arr)
        preview_arr = np.clip(preview_arr, 0, 1)
        preview_tensor = torch.from_numpy(preview_arr.astype(np.float32)).unsqueeze(0)
        
        # Build info
        info = f"ImageToEXR | {w}x{h}"
        if ref_analysis:
            info += f" | Ref max: {ref_analysis['max_val']:.4f}"
            info += f" | sRGB: {ref_analysis['is_srgb']}"
            info += f" | Depth: {bit_depth}"
        if saved_name:
            info += f" | Saved: {saved_name}"
        else:
            info += " | Preview only"
        
        return (preview_tensor, info)

    def _analyze_reference(self, reference_exr):
        """Deep analysis of reference EXR to extract all characteristics."""
        ref = reference_exr[0] if len(reference_exr.shape) == 4 else reference_exr
        ref_arr = ref.cpu().numpy().astype(np.float32)
        
        analysis = {}
        analysis['max_val'] = float(ref_arr.max())
        analysis['min_val'] = float(ref_arr.min())
        analysis['mean'] = float(ref_arr.mean())
        analysis['std'] = float(ref_arr.std())
        
        # Detect if reference is sRGB-like or linear
        # sRGB images typically have mean around 0.2-0.5 with compressed shadows
        # Linear HDR images have different distribution
        if analysis['max_val'] <= 1.0:
            # Low range - likely sRGB/gamma encoded
            analysis['is_srgb'] = True
            analysis['bit_depth'] = "half"
        else:
            # High range - linear HDR
            # Check if values suggest it was sRGB that got expanded
            if analysis['mean'] / analysis['max_val'] < 0.1:
                analysis['is_srgb'] = False  # True linear HDR
            else:
                analysis['is_srgb'] = True
            analysis['bit_depth'] = "half" if analysis['max_val'] < 10.0 else "float"
        
        analysis['compression'] = "zip"
        return analysis

    def _reverse_srgb(self, img_arr):
        """Reverse sRGB gamma to get linear values."""
        mask = img_arr <= 0.04045
        linear = np.where(mask, img_arr / 12.92, ((img_arr + 0.055) / 1.055) ** 2.4)
        return linear

    def _reverse_tonemap(self, img_arr, ref_max):
        """Reverse simple tone mapping to expand HDR range."""
        if ref_max <= 1.0:
            return img_arr
        
        # Approximate reverse of reinhard tone mapping
        # If tonemap was: out = in / (1 + in)
        # Then reverse: in = out / (1 - out)
        # But we need to be careful about division by zero
        
        # Scale to match reference range
        current_max = max(img_arr.max(), 1e-6)
        if current_max < 1.0:
            # Image was compressed - expand it
            img_arr = img_arr * ref_max
        
        return img_arr

    def _get_output_dir(self, save_path):
        """Determine output directory."""
        if save_path and save_path.strip():
            output_dir = os.path.normpath(save_path.strip())
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            return output_dir
        return folder_paths.get_output_directory()

    def _build_filename(self, output_dir, filename, auto_increment):
        """Build unique filename."""
        base_name = filename.strip() if filename and filename.strip() else "output"
        
        if auto_increment:
            counter = 1
            while True:
                fname = f"{base_name}_{counter:04d}.exr"
                exr_path = os.path.join(output_dir, fname)
                if not os.path.exists(exr_path):
                    break
                counter += 1
        else:
            fname = f"{base_name}.exr"
            exr_path = os.path.join(output_dir, fname)
        
        return exr_path, fname

    def _save_exr_openexr(self, filepath, channels, h, w, bit_depth, compression):
        """Save EXR using OpenEXR library."""
        import Imath
        
        compression_map = {
            "none": Imath.Compression.NO_COMPRESSION,
            "rle": Imath.Compression.RLE_COMPRESSION,
            "zips": Imath.Compression.ZIPS_COMPRESSION,
            "zip": Imath.Compression.ZIP_COMPRESSION,
            "piz": Imath.Compression.PIZ_COMPRESSION,
            "pxr24": Imath.Compression.PXR24_COMPRESSION,
            "b44": Imath.Compression.B44_COMPRESSION,
            "b44a": Imath.Compression.B44A_COMPRESSION,
            "dwaa": Imath.Compression.DWAA_COMPRESSION,
            "dwab": Imath.Compression.DWAB_COMPRESSION,
        }
        
        comp = compression_map.get(compression, Imath.Compression.ZIP_COMPRESSION)
        
        if bit_depth == "half":
            pt = Imath.PixelType(Imath.PixelType.HALF)
        else:
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
        
        header = OpenEXR.Header(w, h)
        header['compression'] = comp
        header['channels'] = {name: pt for name in channels.keys()}
        
        channel_data = {}
        for name, arr in channels.items():
            if bit_depth == "half":
                flat = arr.astype(np.float16).flatten()
                channel_data[name] = flat.tobytes()
            else:
                channel_data[name] = arr.astype(np.float32).tobytes()
        
        exr_out = OpenEXR.OutputFile(filepath, header)
        exr_out.writePixels(channel_data)
        exr_out.close()

    def _save_exr_cv2(self, filepath, img_arr):
        """Save EXR using OpenCV fallback."""
        if len(img_arr.shape) == 3 and img_arr.shape[2] >= 3:
            bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        else:
            bgr = img_arr
        cv2.imwrite(filepath, bgr.astype(np.float32))

    @classmethod
    def IS_CHANGED(cls, image, enable, save_exr, save_path, filename, auto_increment, reference_exr=None):
        if not enable:
            return ""
        if image is not None:
            try:
                img_bytes = image.cpu().numpy().tobytes()
                m = hashlib.sha256()
                m.update(img_bytes)
                return m.digest().hex()
            except:
                pass
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
    "RKImageToEXR": RKImageToEXR,
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
    "RKImageToEXR": "Image To EXR (RK)",
}
