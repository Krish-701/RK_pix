import torch
import numpy as np
from PIL import Image, ImageFilter

# ============================================================
# EFFECT NODES
# ============================================================

class RKImageVignette:
    """Add a vignette effect to an image."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "radius": ("FLOAT", {"default": 0.75, "min": 0.1, "max": 1.5, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("vignette_image",)
    CATEGORY = "rk_pix/effect"
    FUNCTION = "apply_vignette"
    
    def apply_vignette(self, image, strength, radius):
        b, h, w, c = image.shape
        device = image.device
        
        # Create coordinate grids
        y = torch.linspace(-1, 1, h, device=device)
        x = torch.linspace(-1, 1, w, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Calculate distance from center
        dist = torch.sqrt(xx**2 + yy**2)
        
        # Create vignette mask
        mask = 1 - torch.clamp((dist - (1 - radius)) / radius, 0, 1)
        mask = mask ** 2  # Smooth falloff
        mask = 1 - (1 - mask) * strength
        
        # Apply to image
        mask = mask.unsqueeze(0).unsqueeze(-1)
        result = image * mask
        
        return (torch.clamp(result, 0, 1),)


class RKImageChromaticAberration:
    """Add chromatic aberration effect."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "red_shift": ("FLOAT", {"default": 2.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "blue_shift": ("FLOAT", {"default": -2.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("aberration_image",)
    CATEGORY = "rk_pix/effect"
    FUNCTION = "apply_aberration"
    
    def apply_aberration(self, image, red_shift, blue_shift, strength):
        b, h, w, c = image.shape
        device = image.device
        
        # Create shifted versions of R and B channels
        result = image.clone()
        
        if abs(red_shift) > 0:
            shift_pixels = int(red_shift)
            if shift_pixels > 0:
                result[:, :, shift_pixels:, 0] = image[:, :, :-shift_pixels, 0]
                result[:, :, :shift_pixels, 0] = image[:, :, :1, 0].expand(-1, -1, shift_pixels, -1)
            elif shift_pixels < 0:
                result[:, :, :shift_pixels, 0] = image[:, :, -shift_pixels:, 0]
                result[:, :, shift_pixels:, 0] = image[:, :, -1:, 0].expand(-1, -1, -shift_pixels, -1)
        
        if abs(blue_shift) > 0:
            shift_pixels = int(blue_shift)
            if shift_pixels > 0:
                result[:, :, shift_pixels:, 2] = image[:, :, :-shift_pixels, 2]
                result[:, :, :shift_pixels, 2] = image[:, :, :1, 2].expand(-1, -1, shift_pixels, -1)
            elif shift_pixels < 0:
                result[:, :, :shift_pixels, 2] = image[:, :, -shift_pixels:, 2]
                result[:, :, shift_pixels:, 2] = image[:, :, -1:, 2].expand(-1, -1, -shift_pixels, -1)
        
        # Blend with original
        result = image * (1 - strength) + result * strength
        
        return (torch.clamp(result, 0, 1),)


class RKImageNoise:
    """Add various types of noise to an image."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_type": (["gaussian", "uniform", "salt_pepper", "perlin"], {"default": "gaussian"}),
                "amount": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("noisy_image",)
    CATEGORY = "rk_pix/effect"
    FUNCTION = "add_noise"
    
    def add_noise(self, image, noise_type, amount, seed):
        torch.manual_seed(seed)
        device = image.device
        
        if noise_type == "gaussian":
            noise = torch.randn_like(image) * amount
            result = image + noise
        elif noise_type == "uniform":
            noise = (torch.rand_like(image) - 0.5) * 2 * amount
            result = image + noise
        elif noise_type == "salt_pepper":
            mask = torch.rand_like(image)
            salt = (mask > (1 - amount / 2)).float()
            pepper = (mask < (amount / 2)).float()
            result = image * (1 - salt - pepper) + salt * 1.0 + pepper * 0.0
        elif noise_type == "perlin":
            noise = self._perlin_noise(image.shape, device) * amount
            result = image + noise
        
        return (torch.clamp(result, 0, 1),)
    
    def _perlin_noise(self, shape, device):
        # Simplified Perlin-like noise
        b, h, w, c = shape
        noise = torch.zeros(shape, device=device)
        
        for octave in range(4):
            freq = 2 ** octave
            amp = 1 / freq
            
            y = torch.linspace(0, freq, h, device=device)
            x = torch.linspace(0, freq, w, device=device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            
            layer = torch.sin(xx * 3.14159) * torch.cos(yy * 3.14159)
            noise += layer.unsqueeze(0).unsqueeze(-1) * amp
        
        return noise


class RKImagePixelate:
    """Pixelate an image."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pixel_size": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pixelated_image",)
    CATEGORY = "rk_pix/effect"
    FUNCTION = "pixelate"
    
    def pixelate(self, image, pixel_size):
        b, h, w, c = image.shape
        
        # Calculate new dimensions
        new_h = h // pixel_size
        new_w = w // pixel_size
        
        if new_h < 1 or new_w < 1:
            return (image,)
        
        # Downscale
        downscaled = comfy.utils.common_upscale(
            image.movedim(-1, 1), new_w, new_h, "area", "center"
        )
        
        # Upscale back with nearest neighbor to keep pixels sharp
        result = comfy.utils.common_upscale(
            downscaled, w, h, "nearest-exact", "center"
        ).movedim(1, -1)
        
        return (result,)


class RKImageGlitch:
    """Add a digital glitch effect."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("glitched_image",)
    CATEGORY = "rk_pix/effect"
    FUNCTION = "glitch"
    
    def glitch(self, image, intensity, seed):
        torch.manual_seed(seed)
        b, h, w, c = image.shape
        device = image.device
        
        result = image.clone()
        
        # Number of glitch bands
        num_bands = int(intensity * 10) + 1
        
        for _ in range(num_bands):
            # Random band position and height
            band_y = torch.randint(0, h - 1, (1,), device=device).item()
            band_h = torch.randint(1, max(1, int(h * 0.1 * intensity)), (1,), device=device).item()
            band_h = min(band_h, h - band_y)
            
            # Random horizontal shift
            shift = torch.randint(-int(w * intensity * 0.5), int(w * intensity * 0.5), (1,), device=device).item()
            
            if shift != 0:
                band = result[:, band_y:band_y+band_h, :, :].clone()
                if shift > 0:
                    result[:, band_y:band_y+band_h, shift:, :] = band[:, :, :-shift, :]
                else:
                    result[:, band_y:band_y+band_h, :shift, :] = band[:, :, -shift:, :]
        
        return (torch.clamp(result, 0, 1),)


class RKImageDuotone:
    """Convert image to duotone effect."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shadow_color": ("STRING", {"default": "#1a1a2e"}),
                "highlight_color": ("STRING", {"default": "#e94560"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("duotone_image",)
    CATEGORY = "rk_pix/effect"
    FUNCTION = "duotone"
    
    def duotone(self, image, shadow_color, highlight_color, threshold):
        # Parse colors
        def parse_color(hex_color):
            hex_color = hex_color.lstrip('#')
            return torch.tensor([
                int(hex_color[0:2], 16) / 255.0,
                int(hex_color[2:4], 16) / 255.0,
                int(hex_color[4:6], 16) / 255.0
            ], device=image.device)
        
        shadow = parse_color(shadow_color)
        highlight = parse_color(highlight_color)
        
        # Convert to luminance
        luminance = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        
        # Create duotone
        t = luminance.unsqueeze(-1)
        result = shadow * (1 - t) + highlight * t
        
        # Apply threshold for more dramatic effect
        if threshold > 0:
            t2 = torch.clamp((t - threshold) / (1 - threshold + 0.001), 0, 1)
            result = shadow * (1 - t2) + highlight * t2
        
        return (torch.clamp(result, 0, 1),)


class RKImagePosterize:
    """Reduce color levels for posterization effect."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "levels": ("INT", {"default": 4, "min": 2, "max": 32, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("posterized_image",)
    CATEGORY = "rk_pix/effect"
    FUNCTION = "posterize"
    
    def posterize(self, image, levels):
        # Quantize to specified levels
        quantized = torch.round(image * (levels - 1)) / (levels - 1)
        return (quantized,)


class RKImageScanlines:
    """Add CRT scanline effect."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "line_height": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "opacity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gap": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("scanline_image",)
    CATEGORY = "rk_pix/effect"
    FUNCTION = "scanlines"
    
    def scanlines(self, image, line_height, opacity, gap):
        b, h, w, c = image.shape
        device = image.device
        
        # Create scanline mask
        mask = torch.ones((h, 1), device=device)
        
        for y in range(h):
            cycle = (y % (line_height + gap))
            if cycle < line_height:
                mask[y, 0] = 1 - opacity
        
        mask = mask.unsqueeze(0).unsqueeze(-1)
        result = image * mask
        
        return (torch.clamp(result, 0, 1),)


class RKImageKaleidoscope:
    """Create a kaleidoscope effect from an image."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "segments": ("INT", {"default": 6, "min": 2, "max": 16, "step": 1}),
                "rotation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("kaleidoscope_image",)
    CATEGORY = "rk_pix/effect"
    FUNCTION = "kaleidoscope"
    
    def kaleidoscope(self, image, segments, rotation):
        b, h, w, c = image.shape
        device = image.device
        
        # Create output
        result = torch.zeros_like(image)
        
        # Center coordinates
        cx, cy = w // 2, h // 2
        
        # Create coordinate grids
        y = torch.arange(h, device=device).float() - cy
        x = torch.arange(w, device=device).float() - cx
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Convert to polar
        angle = torch.atan2(yy, xx) + torch.deg2rad(torch.tensor(rotation, device=device))
        radius = torch.sqrt(xx**2 + yy**2)
        
        # Kaleidoscope mapping
        segment_angle = 2 * np.pi / segments
        angle = torch.abs(torch.remainder(angle, segment_angle) - segment_angle / 2)
        
        # Convert back to cartesian
        new_x = (radius * torch.cos(angle) + cx).long()
        new_y = (radius * torch.sin(angle) + cy).long()
        
        # Clamp coordinates
        new_x = torch.clamp(new_x, 0, w - 1)
        new_y = torch.clamp(new_y, 0, h - 1)
        
        # Sample from original image
        for i in range(b):
            result[i] = image[i, new_y, new_x, :]
        
        return (torch.clamp(result, 0, 1),)


# ============================================================
# REGISTRATION
# ============================================================

EFFECT_CLASS_MAPPINGS = {
    "RKImageVignette": RKImageVignette,
    "RKImageChromaticAberration": RKImageChromaticAberration,
    "RKImageNoise": RKImageNoise,
    "RKImagePixelate": RKImagePixelate,
    "RKImageGlitch": RKImageGlitch,
    "RKImageDuotone": RKImageDuotone,
    "RKImagePosterize": RKImagePosterize,
    "RKImageScanlines": RKImageScanlines,
    "RKImageKaleidoscope": RKImageKaleidoscope,
}

EFFECT_NAME_MAPPINGS = {
    "RKImageVignette": "Vignette (RK)",
    "RKImageChromaticAberration": "Chromatic Aberration (RK)",
    "RKImageNoise": "Add Noise (RK)",
    "RKImagePixelate": "Pixelate (RK)",
    "RKImageGlitch": "Glitch Effect (RK)",
    "RKImageDuotone": "Duotone (RK)",
    "RKImagePosterize": "Posterize (RK)",
    "RKImageScanlines": "Scanlines (RK)",
    "RKImageKaleidoscope": "Kaleidoscope (RK)",
}
