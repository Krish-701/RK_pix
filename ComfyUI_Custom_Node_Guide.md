# ComfyUI Custom Node Development Guide

> **A comprehensive guide for creating, understanding, and publishing ComfyUI custom nodes.**
>
> Based on analysis of real-world node packs: `comfyui_essentials`, `comfyui-impact-pack`, `rgthree-comfy`, `comfyui-kjnodes`
>
> Last Updated: April 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Architecture Overview](#2-architecture-overview)
3. [Project Setup & File Structure](#3-project-setup--file-structure)
4. [Node Properties (The Anatomy of a Node)](#4-node-properties-the-anatomy-of-a-node)
5. [Data Types](#5-data-types)
6. [Input Types & Widgets](#6-input-types--widgets)
7. [Advanced Features](#7-advanced-features)
8. [JavaScript / Frontend Extensions](#8-javascript--frontend-extensions)
9. [Publishing to ComfyUI Registry](#9-publishing-to-comfyui-registry)
10. [Real-World Patterns from Popular Nodes](#10-real-world-patterns-from-popular-nodes)
11. [Complete Example: Image Inverter Node](#11-complete-example-image-inverter-node)
12. [Quick Reference Cheat Sheet](#12-quick-reference-cheat-sheet)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Introduction

### What is a ComfyUI Custom Node?

A **custom node** is a Python class that extends ComfyUI's functionality. Like any Comfy node, it:
- Takes **inputs** (from other nodes or user widgets)
- **Processes** data (images, text, models, etc.)
- Produces **outputs** that can be consumed by other nodes

### Why Create Custom Nodes?

- Implement new image processing algorithms
- Integrate external AI models or APIs
- Add utility functions (batch processing, format conversion, etc.)
- Create reusable workflow components
- Share functionality with the community

### Types of Custom Nodes

| Type | Description | Example |
|------|-------------|---------|
| **Server-side only** | Pure Python, no UI changes | Image processing, model loading |
| **Client-side only** | UI modifications only | Custom widgets, themes |
| **Independent Client + Server** | Both sides, no direct communication | New data type with custom widget |
| **Connected Client + Server** | Direct client-server communication | Real-time preview, external tools |

> ⚠️ **Note:** Nodes requiring client-server communication are **NOT compatible with API mode**.

---

## 2. Architecture Overview

### Client-Server Model

```
┌─────────────────┐         ┌─────────────────┐
│   Client (JS)   │ ◄─────► │  Server (Python)│
│  - UI Rendering │  WebSocket│ - Data Processing│
│  - Node Graph   │         │  - Model Loading │
│  - Widgets      │         │  - Execution     │
└─────────────────┘         └─────────────────┘
```

- **Server (Python)**: Handles all real work — data processing, models, image diffusion
- **Client (JavaScript)**: Handles the user interface — node rendering, connections, widgets
- **API Mode**: Workflows can be sent to the server by non-Comfy clients (CLI, other UIs)

### Execution Flow

```
1. User clicks "Queue Prompt"
2. Client sends workflow JSON to Server
3. Server builds execution graph
4. Server executes nodes (caches outputs for unchanged nodes)
5. Server streams results back to Client
6. Client updates previews/displays results
```

---

## 3. Project Setup & File Structure

### Directory Structure

```
ComfyUI/
├── custom_nodes/
│   └── my_custom_node/           # Your node package
│       ├── __init__.py           # Entry point - registers nodes
│       ├── my_nodes.py           # Node definitions
│       ├── js/                   # Frontend extensions (optional)
│       │   └── my_extension.js
│       ├── pyproject.toml        # Package metadata (for registry)
│       └── requirements.txt      # Python dependencies (optional)
```

### Minimal `__init__.py`

```python
"""
@author: Your Name
@title: My Custom Node
@nickname: my-node
@description: A brief description of what this node does.
"""

from .my_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
```

### Node Registration

Every node file must define two dictionaries:

```python
# Maps internal class names to node classes
NODE_CLASS_MAPPINGS = {
    "MyNodeClass": MyNodeClass,
    "AnotherNode": AnotherNode,
}

# Maps internal names to display names (shown in UI)
NODE_DISPLAY_NAME_MAPPINGS = {
    "MyNodeClass": "My Node",
    "AnotherNode": "Another Node",
}
```

---

## 4. Node Properties (The Anatomy of a Node)

### Minimal Node Example

```python
class InvertImageNode:
    """A simple node that inverts an image."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_in": ("IMAGE", {}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    CATEGORY = "examples"
    FUNCTION = "invert"
    
    def invert(self, image_in):
        image_out = 1.0 - image_in
        return (image_out,)
```

### Required Properties

| Property | Type | Description |
|----------|------|-------------|
| `INPUT_TYPES` | `@classmethod` | Defines all inputs (required, optional, hidden) |
| `RETURN_TYPES` | `tuple[str]` | Data types of outputs |
| `CATEGORY` | `str` | Location in Add Node menu (use `/` for submenus) |
| `FUNCTION` | `str` | Name of the method to call when executing |

### Optional Properties

| Property | Type | Description |
|----------|------|-------------|
| `RETURN_NAMES` | `tuple[str]` | Human-readable output labels |
| `OUTPUT_NODE` | `bool` | Set `True` if this is an output node (always executes) |
| `IS_CHANGED` | `@classmethod` | Controls when node re-executes (caching) |
| `VALIDATE_INPUTS` | `@classmethod` | Pre-execution input validation |
| `SEARCH_ALIASES` | `list[str]` | Alternative names for search |

---

## 5. Data Types

### Primitive Types

| Type | Python Type | Widget | Parameters |
|------|-------------|--------|------------|
| `INT` | `int` | Number input / slider | `default`, `min`, `max`, `step` |
| `FLOAT` | `float` | Number input / slider | `default`, `min`, `max`, `step`, `round` |
| `STRING` | `str` | Text input | `multiline`, `placeholder`, `dynamicPrompts` |
| `BOOLEAN` | `bool` | Toggle | `label_on`, `label_off` |
| `COMBO` | `str` | Dropdown | Defined as `list[str]` of options |

### Tensor / Media Types

| Type | Python Type | Shape / Format |
|------|-------------|----------------|
| `IMAGE` | `torch.Tensor` | `[B, H, W, C]` (batch, height, width, channels) |
| `LATENT` | `dict` | Contains `samples`: `[B, C, H, W]` |
| `MASK` | `torch.Tensor` | `[H, W]` or `[B, C, H, W]` |
| `AUDIO` | `dict` | Contains `waveform`: `[B, C, T]`, `sample_rate` |

### Model / Diffusion Types

| Type | Description |
|------|-------------|
| `MODEL` | Diffusion model object |
| `CLIP` | CLIP text encoder model |
| `VAE` | Variational Autoencoder |
| `CONDITIONING` | Text conditioning data |
| `NOISE` | Noise generator object (has `generate_noise()` method) |
| `SAMPLER` | Sampling object (has `sample()` method) |
| `SIGMAS` | 1D tensor of sigma values `[steps+1]` |
| `GUIDER` | Callable that guides denoising |

### Custom Types

You can define your own data types! Just use any string:

```python
# Define a custom type
"my_custom_data": ("MY_CUSTOM_TYPE", {}),

# Output it
RETURN_TYPES = ("MY_CUSTOM_TYPE",)
```

The frontend will only allow connections between matching types.

### The "Any" Type Pattern (from pythongosssss / kjnodes)

```python
class AnyType(str):
    """A type that accepts any connection."""
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

# Usage in INPUT_TYPES
"input": (any_type, {})
```

---

## 6. Input Types & Widgets

### INPUT_TYPES Structure

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {    # Must be connected or have a value
            "input_name": ("TYPE", {parameters}),
        },
        "optional": {    # Can be left unconnected
            "optional_input": ("TYPE", {parameters}),
        },
        "hidden": {      # Not shown as sockets, passed automatically
            "prompt": "PROMPT",      # Full workflow JSON
            "extra_pnginfo": "EXTRA_PNGINFO",  # Metadata
            "unique_id": "UNIQUE_ID",  # Node's unique ID
        },
    }
```

### Number Inputs (INT / FLOAT)

```python
"steps": ("INT", {
    "default": 20,
    "min": 1,
    "max": 100,
    "step": 1,        # For slider increments
}),

"strength": ("FLOAT", {
    "default": 1.0,
    "min": 0.0,
    "max": 2.0,
    "step": 0.01,
    "round": 0.001,   # Precision (defaults to step)
}),
```

### String Inputs

```python
"prompt": ("STRING", {
    "default": "a beautiful landscape",
    "multiline": True,        # True = multi-line text area
    "placeholder": "Enter prompt...",
    "dynamicPrompts": True,   # Enable dynamic prompt syntax
}),
```

### Boolean Inputs

```python
"enabled": ("BOOLEAN", {
    "default": True,
    "label_on": "Enabled",
    "label_off": "Disabled",
}),
```

### Combo (Dropdown) Inputs

```python
# Static options
"mode": (["fast", "quality", "balanced"], {
    "default": "balanced",
}),

# Dynamic options (computed at runtime)
"ckpt_name": (folder_paths.get_filename_list("checkpoints"), {}),
```

### Force Input Socket (No Widget)

```python
# Shows as input socket only, no widget
"image": ("IMAGE", {
    "forceInput": True,
}),

# Defaults to input socket but can be converted to widget
"value": ("FLOAT", {
    "defaultInput": True,
}),
```

---

## 7. Advanced Features

### 7.1 Caching & IS_CHANGED

ComfyUI caches node outputs and only re-executes when inputs change.

```python
class LoadImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("STRING", {})},
        }
    
    # Override to detect external file changes
    @classmethod
    def IS_CHANGED(cls, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()  # Returns hash; re-executes if hash changes
    
    def load_image(self, image):
        # ... load and return image
        return (image_tensor,)
```

**Important:** `IS_CHANGED` should **NOT** return `bool`!
- Return `float("NaN")` to always re-execute
- Return a hash, timestamp, or any comparable object

### 7.2 Lazy Evaluation

Skip evaluating inputs that aren't needed:

```python
class MixImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE", {"lazy": True}),
                "image2": ("IMAGE", {"lazy": True}),
                "mask": ("MASK",),
            },
        }
    
    # Called before execution to determine which lazy inputs are needed
    def check_lazy_status(self, mask, image1, image2):
        mask_min = mask.min()
        mask_max = mask.max()
        needed = []
        
        if image1 is None and mask_max != 0.0:
            needed.append("image1")
        if image2 is None and mask_min != 1.0:
            needed.append("image2")
        
        return needed
    
    def mix(self, mask, image1, image2):
        if mask.max() == 0.0:
            return (image1,)
        if mask.min() == 1.0:
            return (image2,)
        result = image1 * (1.0 - mask) + image2 * mask
        return (result,)
```

### 7.3 Execution Blocking

Block downstream execution conditionally:

```python
from comfy_execution.graph import ExecutionBlocker

class ConditionalOutput:
    def process(self, image, enabled):
        if not enabled:
            # Silently block execution
            return (ExecutionBlocker(None),)
        
        # Or block with an error message
        if image is None:
            return (ExecutionBlocker("No image provided!"),)
        
        return (image,)
```

### 7.4 Input Validation

```python
class AddNumbers:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input1": ("INT,FLOAT", {"min": 0, "max": 1000}),
                "input2": ("INT,FLOAT", {"min": 0, "max": 1000}),
            },
        }
    
    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # input_types dict: {input_name: connected_output_type}
        if input_types["input1"] not in ("INT", "FLOAT"):
            return "input1 must be an INT or FLOAT type"
        if input_types["input2"] not in ("INT", "FLOAT"):
            return "input2 must be an INT or FLOAT type"
        return True
```

### 7.5 Hidden Inputs

Hidden inputs are automatically populated by ComfyUI:

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {"text": ("STRING", {})},
        "hidden": {
            "prompt": "PROMPT",           # Full workflow JSON
            "extra_pnginfo": "EXTRA_PNGINFO",  # PNG metadata
            "unique_id": "UNIQUE_ID",     # Node's unique ID in graph
        },
    }

def process(self, text, prompt, extra_pnginfo, unique_id):
    # Access workflow data
    print(f"Node ID: {unique_id}")
    print(f"Workflow: {prompt}")
    return (text,)
```

---

## 8. JavaScript / Frontend Extensions

### Basic Frontend Extension

Create a `js/` folder in your custom node directory:

```javascript
// js/my_extension.js
import { app } from "../../scripts/app.js";

// Register extension
app.registerExtension({
    name: "MyNode.Extension",
    
    // Called before node is created
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "MyNodeClass") {
            // Modify node behavior
        }
    },
    
    // Called when node is created
    async nodeCreated(node) {
        if (node.comfyClass === "MyNodeClass") {
            // Add custom UI elements
        }
    },
});
```

### Custom Widgets

```javascript
// Add a custom color picker widget
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "MyNode.ColorPicker",
    
    async getCustomWidgets(app) {
        return {
            COLOR(widget, node) {
                // Create custom widget element
                const input = document.createElement("input");
                input.type = "color";
                input.value = widget.value || "#ffffff";
                
                input.addEventListener("input", (e) => {
                    widget.value = e.target.value;
                });
                
                return { widget: input };
            }
        };
    }
});
```

### Custom API Routes (Server-side)

```python
from aiohttp import web
from server import PromptServer

@PromptServer.instance.routes.get("/my_node/hello")
async def get_hello(request):
    return web.json_response({"message": "Hello from custom node!"})

@PromptServer.instance.routes.post("/my_node/process")
async def post_process(request):
    data = await request.json()
    result = {"processed": data.get("input", "")}
    return web.json_response(result)
```

---

## 9. Publishing to ComfyUI Registry

### `pyproject.toml` Configuration

```toml
[project]
name = "my-custom-node"           # Unique node ID (no "ComfyUI" prefix)
version = "1.0.0"                 # Semantic versioning
license = {file = "LICENSE"}      # Or: license = {text = "MIT"}
description = "Does amazing image processing"
repository = "https://github.com/username/my-custom-node"
urls = { Documentation = "https://docs.example.com" }
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.21.0",
    "pillow>=9.0.0",
]

[project.optional-dependencies]
dev = ["pytest", "black"]

[tool.comfy]
PublisherId = "your-github-username"
DisplayName = "My Amazing Node"
Icon = "https://example.com/icon.svg"      # 400x400, SVG/PNG/JPG/GIF
Banner = "https://example.com/banner.png"  # 21:9 aspect ratio
requires-comfyui = ">=0.3.0"
```

### Publishing Steps

1. **Create GitHub Repository**
   - Push your code with `pyproject.toml`

2. **Register as Publisher**
   - Go to [registry.comfy.org](https://registry.comfy.org)
   - Sign in with GitHub
   - Create publisher profile

3. **Publish Node**
   ```bash
   # Install comfy-cli
   pip install comfy-cli
   
   # Login
   comfy node login
   
   # Publish (from repo root)
   comfy node publish
   ```

4. **Install via ComfyUI-Manager**
   - Your node will appear in the Manager's node list
   - Users can install with one click

---

## 10. Real-World Patterns from Popular Nodes

### 10.1 __init__.py Patterns

#### Pattern A: Modular Merge (comfyui_essentials)

```python
# image.py
IMAGE_CLASS_MAPPINGS = {"ImageResize": ImageResize}
IMAGE_NAME_MAPPINGS = {"ImageResize": "Image Resize"}

# mask.py
MASK_CLASS_MAPPINGS = {"MaskToImage": MaskToImage}
MASK_NAME_MAPPINGS = {"MaskToImage": "Mask To Image"}

# __init__.py
from .image import IMAGE_CLASS_MAPPINGS, IMAGE_NAME_MAPPINGS
from .mask import MASK_CLASS_MAPPINGS, MASK_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(IMAGE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(IMAGE_NAME_MAPPINGS)
NODE_CLASS_MAPPINGS.update(MASK_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(MASK_NAME_MAPPINGS)
```

#### Pattern B: Direct Dictionary (comfyui-impact-pack)

```python
NODE_CLASS_MAPPINGS = {
    "SAMLoader": SAMLoader,
    "DetailerForEach": DetailerForEach,
    # ... 200+ entries
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAMLoader": "SAMLoader (Impact)",
    # ...
}
```

#### Pattern C: Class NAME Attribute (rgthree-comfy)

```python
class RgthreeContext:
    NAME = get_name("Context")  # "Context (rgthree)"
    
NODE_CLASS_MAPPINGS = {
    RgthreeContext.NAME: RgthreeContext,
}
```

#### Pattern D: NODE_CONFIG + Generator (comfyui-kjnodes)

```python
NODE_CONFIG = {
    "BOOLConstant": {"class": BOOLConstant, "name": "BOOL Constant"},
    "INTConstant": {"class": INTConstant, "name": "INT Constant"},
}

def generate_node_mappings(node_config):
    mappings = {}
    display_names = {}
    for name, info in node_config.items():
        mappings[name] = info["class"]
        display_names[name] = info.get("name", info["class"].__name__)
    return mappings, display_names

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)
```

### 10.2 IMAGE Tensor Handling

Standard shape: `(B, H, W, C)` where:
- `B` = batch size
- `H` = height
- `W` = width
- `C` = channels (usually 3 for RGB)

```python
# Permute for comfy utils: B,H,W,C -> B,C,H,W
image.permute([0, 3, 1, 2])

# Or using movedim (cleaner)
image.movedim(-1, 1)  # Last dim to position 1

# Common upscale pattern
comfy.utils.common_upscale(
    image.movedim(-1, 1),  # B,H,W,C -> B,C,H,W
    width, height,
    method,  # "nearest", "bilinear", "area", "bicubic", "lanczos"
    crop="disabled"  # or "center"
).movedim(1, -1)  # Back to B,H,W,C

# Clamp values
image = torch.clamp(image, 0.0, 1.0)

# IMAGE -> PIL
pil_image = Image.fromarray(
    np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
)

# PIL -> IMAGE
tensor = torch.from_numpy(
    np.array(pil_image).astype(np.float32) / 255.0
).unsqueeze(0)  # Add batch dimension
```

### 10.3 MASK Handling

```python
# Mask shapes: [H,W] or [B,1,H,W] or [B,H,W]
# For operations, often need to expand:
mask_expanded = mask.unsqueeze(-1)  # [H,W] -> [H,W,1]

# Apply mask to image
result = image * mask_expanded

# Invert mask
inverted = 1.0 - mask
```

### 10.4 Conditional Import Pattern

```python
try:
    import cv2
except ImportError:
    print("[MyNode] OpenCV not installed, some features disabled")
    cv2 = None

def process_with_cv2(image):
    if cv2 is None:
        raise Exception("OpenCV is required for this operation")
    # ... use cv2
```

### 10.5 Model Patching Pattern (kjnodes)

```python
class PatchAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    
    def patch(self, model, enable):
        model_clone = model.clone()
        
        def patch_attention(model):
            # Modify model internals
            pass
        
        if enable:
            model_clone.add_callback(CallbacksMP.ON_PRE_RUN, patch_attention)
        
        return (model_clone,)
```

### 10.6 Display/Output Node Pattern

```python
class DisplayAny:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"source": (any_type, {})},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    RETURN_TYPES = ()
    FUNCTION = "main"
    OUTPUT_NODE = True  # Critical for display nodes
    
    def main(self, source=None, unique_id=None, extra_pnginfo=None):
        # Serialize value for display
        value = str(source)
        
        # Return UI data - this gets sent to frontend
        return {"ui": {"text": (value,)}}
```

### 10.7 Frontend Extension: Display Value (rgthree)

```javascript
import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "rgthree.DisplayAny",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Display Any (rgthree)") {
            
            // Store original onNodeCreated
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            // Override to add custom widget
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, []);
                
                this.showValueWidget = ComfyWidgets["STRING"](
                    this, "output", ["STRING", { multiline: true }], app
                ).widget;
                
                this.showValueWidget.inputEl.readOnly = true;
            };
            
            // Handle execution results
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, [message]);
                this.showValueWidget.value = message.text[0];
            };
        }
    },
});
```

### 10.8 Frontend Extension: Custom DOM Widget (kjnodes)

```javascript
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "MyNode.SplineEditor",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Spline Editor") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                
                // Create custom DOM element
                const container = document.createElement("div");
                container.style.width = "300px";
                container.style.height = "200px";
                
                // Add canvas or other UI
                const canvas = document.createElement("canvas");
                container.appendChild(canvas);
                
                // Register as DOM widget
                const widget = this.addDOMWidget("editor", "custom", container);
                widget.serialize = false;  // Don't save in workflow
            };
        }
    },
});
```

### 10.9 Server API Route Pattern

```python
from aiohttp import web
from server import PromptServer

# GET endpoint
@PromptServer.instance.routes.get("/my_node/status")
async def get_status(request):
    return web.json_response({"status": "ok", "version": "1.0.0"})

# POST endpoint
@PromptServer.instance.routes.post("/my_node/process")
async def post_process(request):
    data = await request.json()
    result = process_data(data)
    return web.json_response({"result": result})

# Static file serving
import os
from pathlib import Path

PromptServer.instance.app.add_routes([
    web.static("/my_node_assets", 
               (Path(__file__).parent / "assets").as_posix())
])
```

### 10.10 Error Handling Patterns

```python
# Pattern 1: Hard fail with descriptive message
if len(image.shape) != 4:
    raise Exception(f"[MyNode] Expected 4D tensor [B,H,W,C], got {image.shape}")

# Pattern 2: Batch size check
if image.shape[0] > 1:
    raise Exception("[MyNode] Batch processing not supported for this operation")

# Pattern 3: Graceful fallback
try:
    import optional_dependency
except ImportError:
    print("[MyNode] Warning: optional_dependency not found, using fallback")
    optional_dependency = None
```

---

## 11. Complete Example: Image Inverter Node

### File: `__init__.py`

```python
"""
@author: Your Name
@title: Image Inverter
@nickname: img-invert
@description: A simple node that inverts image colors.
"""

from .image_inverter import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
```

### File: `image_inverter.py`

```python
import torch
import numpy as np
from PIL import Image

class ImageInverter:
    """Inverts the colors of an image."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
            },
            "optional": {
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("inverted_image",)
    CATEGORY = "image/processing"
    FUNCTION = "invert"
    
    def invert(self, image, strength=1.0):
        """
        Invert image colors.
        
        Args:
            image: torch.Tensor [B, H, W, C] in range [0, 1]
            strength: How much to invert (0 = original, 1 = fully inverted)
        
        Returns:
            Inverted image tensor
        """
        # image is [B, H, W, C], values in [0, 1]
        inverted = 1.0 - image
        result = image * (1.0 - strength) + inverted * strength
        return (result,)

class ImageChannelSplitter:
    """Split an image into RGB channels."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("red", "green", "blue")
    CATEGORY = "image/processing"
    FUNCTION = "split_channels"
    
    def split_channels(self, image):
        """Split image into R, G, B channels."""
        # image: [B, H, W, C]
        r = image.clone()
        r[:, :, :, 1] = 0  # Zero G
        r[:, :, :, 2] = 0  # Zero B
        
        g = image.clone()
        g[:, :, :, 0] = 0  # Zero R
        g[:, :, :, 2] = 0  # Zero B
        
        b = image.clone()
        b[:, :, :, 0] = 0  # Zero R
        b[:, :, :, 1] = 0  # Zero G
        
        return (r, g, b)

# Register nodes
NODE_CLASS_MAPPINGS = {
    "ImageInverter": ImageInverter,
    "ImageChannelSplitter": ImageChannelSplitter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageInverter": "Invert Image",
    "ImageChannelSplitter": "Split RGB Channels",
}
```

### File: `pyproject.toml`

```toml
[project]
name = "image-inverter"
version = "1.0.0"
license = {text = "MIT"}
description = "Simple image inversion and channel splitting nodes for ComfyUI"
repository = "https://github.com/username/comfyui-image-inverter"
requires-python = ">=3.9"

[tool.comfy]
PublisherId = "your-github-username"
DisplayName = "Image Inverter"
```

---

## 12. Quick Reference Cheat Sheet

### Node Template

```python
class MyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {},
            "hidden": {},
        }
    
    RETURN_TYPES = ("TYPE",)
    RETURN_NAMES = ("name",)
    CATEGORY = "category/subcategory"
    FUNCTION = "execute"
    OUTPUT_NODE = False
    
    def execute(self, **kwargs):
        return (result,)
```

### Common Patterns

```python
# Multiple outputs
RETURN_TYPES = ("IMAGE", "MASK", "STRING")
RETURN_NAMES = ("image", "mask", "info")

# Subcategory
CATEGORY = "image/transforms"

# Always re-execute
@classmethod
def IS_CHANGED(cls, **kwargs):
    return float("NaN")

# Optional with default
def execute(self, input1, optional_input=None):
    pass

# Accept any type with **kwargs
def execute(self, **kwargs):
    for name, value in kwargs.items():
        print(f"{name}: {value}")
```

### Input Parameters Quick Reference

| Parameter | Types | Description |
|-----------|-------|-------------|
| `default` | All | Default value |
| `min` | INT, FLOAT | Minimum value |
| `max` | INT, FLOAT | Maximum value |
| `step` | INT, FLOAT | Increment step |
| `round` | FLOAT | Rounding precision |
| `multiline` | STRING | Multi-line text area |
| `placeholder` | STRING | Placeholder text |
| `dynamicPrompts` | STRING | Enable dynamic prompt syntax |
| `label_on` | BOOLEAN | Label when True |
| `label_off` | BOOLEAN | Label when False |
| `defaultInput` | All | Default to input socket |
| `forceInput` | All | Force input socket (no widget) |
| `lazy` | All | Enable lazy evaluation |
| `rawLink` | All | Receive link instead of value |
| `tooltip` | All | Hover help text |

---

## 13. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Node not appearing | Check `__init__.py` exports `NODE_CLASS_MAPPINGS` |
| Import errors | Verify `requirements.txt` or `pyproject.toml` dependencies |
| Type mismatch | Ensure `RETURN_TYPES` is a tuple with trailing comma for single output |
| Caching issues | Implement `IS_CHANGED` to detect external changes |
| Frontend not loading | Check `WEB_DIRECTORY` path and JS syntax |

### Debug Tips

```python
# Print to server console
print(f"Debug: {variable}")

# Check node info via API
curl http://localhost:8188/object_info/MyNodeClass

# Enable verbose logging
# Start ComfyUI with: --verbose
```

### Best Practices

1. ✅ **Always use tuples** for `RETURN_TYPES` — even for single outputs: `("IMAGE",)`
2. ✅ **Provide `RETURN_NAMES`** for better UX in the node graph
3. ✅ **Use lazy evaluation** when inputs might not be needed
4. ✅ **Implement `IS_CHANGED`** for nodes that load external files
5. ✅ **Validate inputs** with `VALIDATE_INPUTS` for early error detection
6. ✅ **Use semantic versioning** in `pyproject.toml`
7. ✅ **Document your nodes** with docstrings
8. ✅ **Use `[PackName]` prefix** in error messages for clarity
9. ✅ **Handle batch dimensions** properly — always expect `[B, H, W, C]`
10. ❌ **Don't return `True` from `IS_CHANGED`** — it breaks caching
11. ❌ **Don't assume directory names** for imports (use relative imports)
12. ❌ **Don't require client-server communication** if you want API compatibility

---

## Resources

- [Official ComfyUI Docs](https://docs.comfy.org/)
- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI Registry](https://registry.comfy.org/)
- [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
- [Example Node](https://github.com/comfyanonymous/ComfyUI/blob/master/custom_nodes/example_node.py.example)

---

> **Happy node building!** 🚀
