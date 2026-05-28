import torch
import numpy as np
import hashlib
import json

# ============================================================
# UTILITY NODES
# ============================================================

class AnyType(str):
    """A type that accepts any connection."""
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")


class RKNumberMath:
    """Perform math operations on numbers."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operation": (["add", "subtract", "multiply", "divide", "power", "modulo", "min", "max", "abs"], {"default": "add"}),
                "a": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "b": ("FLOAT", {"default": 0.0, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("float_result", "int_result")
    CATEGORY = "rk_pix/utility"
    FUNCTION = "calculate"
    
    def calculate(self, operation, a, b):
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            result = a / b if b != 0 else 0
        elif operation == "power":
            result = a ** b
        elif operation == "modulo":
            result = a % b if b != 0 else 0
        elif operation == "min":
            result = min(a, b)
        elif operation == "max":
            result = max(a, b)
        elif operation == "abs":
            result = abs(a)
        
        return (float(result), int(result))


class RKNumberRemap:
    """Remap a number from one range to another."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.5, "step": 0.01}),
                "old_min": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "old_max": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "new_min": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "new_max": ("FLOAT", {"default": 100.0, "step": 0.01}),
                "clamp": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("remapped_value",)
    CATEGORY = "rk_pix/utility"
    FUNCTION = "remap"
    
    def remap(self, value, old_min, old_max, new_min, new_max, clamp):
        if old_max == old_min:
            return (new_min,)
        
        result = (value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
        
        if clamp:
            result = max(new_min, min(new_max, result))
        
        return (result,)


class RKStringConcat:
    """Concatenate strings with a separator."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "separator": ("STRING", {"default": " "}),
            },
            "optional": {
                "string1": ("STRING", {"default": ""}),
                "string2": ("STRING", {"default": ""}),
                "string3": ("STRING", {"default": ""}),
                "string4": ("STRING", {"default": ""}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("concatenated",)
    CATEGORY = "rk_pix/utility"
    FUNCTION = "concat"
    
    def concat(self, separator, string1="", string2="", string3="", string4=""):
        strings = [s for s in [string1, string2, string3, string4] if s]
        result = separator.join(strings)
        return (result,)


class RKStringReplace:
    """Replace text in a string."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "find": ("STRING", {"default": ""}),
                "replace": ("STRING", {"default": ""}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    CATEGORY = "rk_pix/utility"
    FUNCTION = "replace_text"
    
    def replace_text(self, text, find, replace):
        result = text.replace(find, replace)
        return (result,)


class RKImageInfo:
    """Get information about an image tensor."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("batch", "height", "width", "info")
    CATEGORY = "rk_pix/utility"
    FUNCTION = "get_info"
    
    def get_info(self, image):
        b, h, w, c = image.shape
        info = f"Shape: [{b}, {h}, {w}, {c}] | Range: [{image.min():.4f}, {image.max():.4f}] | Mean: {image.mean():.4f}"
        return (b, h, w, info)


class RKImageToMask:
    """Convert an image to a mask using luminance."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    CATEGORY = "rk_pix/utility"
    FUNCTION = "to_mask"
    
    def to_mask(self, image):
        # Convert RGB to luminance
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return (luminance,)


class RKMaskToImage:
    """Convert a mask to a grayscale image."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "rk_pix/utility"
    FUNCTION = "to_image"
    
    def to_image(self, mask):
        # Expand mask to 3 channels
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(-1)
        elif len(mask.shape) == 3:
            mask = mask.unsqueeze(-1)
        
        image = mask.repeat(1, 1, 1, 3)
        return (image,)


class RKImageHash:
    """Generate a hash of an image for change detection."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("hash",)
    CATEGORY = "rk_pix/utility"
    FUNCTION = "get_hash"
    
    def get_hash(self, image):
        # Convert to bytes and hash
        img_bytes = (image.cpu().numpy() * 255).astype(np.uint8).tobytes()
        hash_value = hashlib.md5(img_bytes).hexdigest()
        return (hash_value,)
    
    @classmethod
    def IS_CHANGED(cls, image):
        img_bytes = (image.cpu().numpy() * 255).astype(np.uint8).tobytes()
        return hashlib.md5(img_bytes).hexdigest()


class RKBatchSelector:
    """Select a specific image from a batch."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "index": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("selected_image",)
    CATEGORY = "rk_pix/utility"
    FUNCTION = "select"
    
    def select(self, image, index):
        b = image.shape[0]
        if b == 1:
            return (image,)
        
        idx = min(index, b - 1)
        return (image[idx:idx+1],)


class RKImageListToBatch:
    """Combine multiple images into a batch."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("batch",)
    CATEGORY = "rk_pix/utility"
    FUNCTION = "combine"
    
    def combine(self, image1, image2=None, image3=None, image4=None):
        images = [image1]
        
        for img in [image2, image3, image4]:
            if img is not None:
                # Resize to match first image if needed
                if img.shape[1:3] != image1.shape[1:3]:
                    h, w = image1.shape[1:3]
                    img = comfy.utils.common_upscale(
                        img.movedim(-1, 1), w, h, "bilinear", "center"
                    ).movedim(1, -1)
                images.append(img)
        
        return (torch.cat(images, dim=0),)


# ============================================================
# REGISTRATION
# ============================================================

UTILITY_CLASS_MAPPINGS = {
    "RKNumberMath": RKNumberMath,
    "RKNumberRemap": RKNumberRemap,
    "RKStringConcat": RKStringConcat,
    "RKStringReplace": RKStringReplace,
    "RKImageInfo": RKImageInfo,
    "RKImageToMask": RKImageToMask,
    "RKMaskToImage": RKMaskToImage,
    "RKImageHash": RKImageHash,
    "RKBatchSelector": RKBatchSelector,
    "RKImageListToBatch": RKImageListToBatch,
}

UTILITY_NAME_MAPPINGS = {
    "RKNumberMath": "Number Math (RK)",
    "RKNumberRemap": "Number Remap (RK)",
    "RKStringConcat": "String Concat (RK)",
    "RKStringReplace": "String Replace (RK)",
    "RKImageInfo": "Image Info (RK)",
    "RKImageToMask": "Image To Mask (RK)",
    "RKMaskToImage": "Mask To Image (RK)",
    "RKImageHash": "Image Hash (RK)",
    "RKBatchSelector": "Batch Selector (RK)",
    "RKImageListToBatch": "Images To Batch (RK)",
}
