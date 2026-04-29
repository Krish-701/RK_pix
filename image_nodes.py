import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import comfy.utils

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
# REGISTRATION
# ============================================================

IMAGE_CLASS_MAPPINGS = {
    "RKImageBlend": RKImageBlend,
    "RKImageColorAdjust": RKImageColorAdjust,
    "RKImageTile": RKImageTile,
    "RKImageMirror": RKImageMirror,
    "RKImageCrop": RKImageCrop,
    "RKImageBorder": RKImageBorder,
}

IMAGE_NAME_MAPPINGS = {
    "RKImageBlend": "Blend Images (RK)",
    "RKImageColorAdjust": "Color Adjust (RK)",
    "RKImageTile": "Tile Image (RK)",
    "RKImageMirror": "Mirror Image (RK)",
    "RKImageCrop": "Crop Image (RK)",
    "RKImageBorder": "Add Border (RK)",
}
