import torch
import numpy as np
from PIL import Image
import os
import folder_paths
import hashlib

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[RK_VideoNodes] Warning: OpenCV not available. Video features disabled.")


# ============================================================
# RK VIDEO LOADER - Universal Video/Image/Sequence Loader
# ============================================================

class RK_VideoLoader:
    """
    Universal loader for videos, image sequences, and single images.
    Outputs both IMAGE batch and VIDEO format for compatibility.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif']
        all_extensions = video_extensions + image_extensions
        
        files = []
        if os.path.exists(input_dir):
            for f in sorted(os.listdir(input_dir)):
                if any(f.lower().endswith(ext) for ext in all_extensions):
                    files.append(f)
        
        return {
            "required": {
                "video": (sorted(files),),
                "force_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01}),
                "force_size": (["Disabled", "Custom Width/Height", "256x256", "512x512", "768x768", "1024x1024", "1280x720", "1920x1080"], {"default": "Disabled"}),
                "custom_width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "custom_height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "end_frame": ("INT", {"default": -1, "min": -1, "max": 99999, "step": 1}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    RETURN_TYPES = ("IMAGE", "VIDEO", "INT", "FLOAT", "INT", "STRING", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("images", "video", "frame_count", "fps", "resolution", 
                    "video_info", "duration", "current_frame", "total_frames")
    CATEGORY = "rk_pix/video"
    FUNCTION = "load_video"
    
    def load_video(self, video, force_rate, force_size, custom_width, custom_height,
                   start_frame, end_frame, frame_load_cap, select_every_nth,
                   prompt=None, unique_id=None, extra_pnginfo=None):
        
        input_dir = folder_paths.get_input_directory()
        video_path = os.path.join(input_dir, video)
        
        if not os.path.exists(video_path):
            raise Exception(f"[RK_VideoLoader] File not found: {video_path}")
        
        ext = os.path.splitext(video)[1].lower()
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        is_video = ext in video_extensions
        
        if is_video:
            frames, fps, total_frames, duration = self._load_video_file(
                video_path, force_rate, force_size, custom_width, custom_height,
                start_frame, end_frame, frame_load_cap, select_every_nth
            )
        else:
            frames, fps, total_frames, duration = self._load_single_image(
                video_path, force_size, custom_width, custom_height
            )
        
        if len(frames) > 1:
            info = f"Video: {video} | Frames: {len(frames)}/{total_frames} | FPS: {fps:.2f} | Duration: {duration:.2f}s | Size: {frames[0].shape[1]}x{frames[0].shape[0]}"
        else:
            info = f"Image: {video} | Size: {frames[0].shape[1]}x{frames[0].shape[0]}"
        
        # Stack frames into batch tensor [B, H, W, C]
        if len(frames) == 1:
            images = frames[0].unsqueeze(0)
        else:
            images = torch.stack(frames)
        
        # VIDEO output is same tensor but marked differently for nodes that expect video
        video_out = images.clone()
        
        resolution = frames[0].shape[1] * frames[0].shape[0]
        
        return (images, video_out, len(frames), fps, resolution, info, duration, 0, total_frames)
    
    def _load_video_file(self, video_path, force_rate, force_size, custom_width, 
                         custom_height, start_frame, end_frame, frame_load_cap, select_every_nth):
        
        if not CV2_AVAILABLE:
            raise Exception("[RK_VideoLoader] OpenCV required. Install: pip install opencv-python")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception(f"[RK_VideoLoader] Cannot open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / original_fps if original_fps > 0 else 0
        
        fps = force_rate if force_rate > 0 else original_fps
        target_w, target_h = self._get_target_size(force_size, custom_width, custom_height, width, height)
        
        effective_start = max(0, start_frame)
        effective_end = end_frame if end_frame >= 0 else total_frames
        effective_end = min(effective_end, total_frames)
        
        frames_to_load = []
        loaded_count = 0
        
        if force_rate > 0 and original_fps > 0:
            frame_step = original_fps / force_rate
        else:
            frame_step = 1.0
        
        current_step = 0.0
        frame_idx = effective_start
        
        while frame_idx < effective_end:
            if frame_load_cap > 0 and loaded_count >= frame_load_cap:
                break
            
            if (frame_idx - effective_start) % select_every_nth != 0:
                frame_idx += 1
                continue
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if target_w != width or target_h != height:
                frame_rgb = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            
            tensor = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
            frames_to_load.append(tensor)
            
            loaded_count += 1
            current_step += frame_step
            frame_idx = effective_start + int(current_step)
        
        cap.release()
        
        if not frames_to_load:
            raise Exception("[RK_VideoLoader] No frames could be loaded")
        
        return frames_to_load, fps, total_frames, duration
    
    def _load_single_image(self, image_path, force_size, custom_width, custom_height):
        img = Image.open(image_path)
        img = img.convert('RGB')
        
        width, height = img.size
        target_w, target_h = self._get_target_size(force_size, custom_width, custom_height, width, height)
        
        if target_w != width or target_h != height:
            img = img.resize((target_w, target_h), Image.LANCZOS)
        
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)
        
        return [tensor], 1.0, 1, 0.0
    
    def _get_target_size(self, force_size, custom_width, custom_height, src_w, src_h):
        size_map = {
            "256x256": (256, 256), "512x512": (512, 512), "768x768": (768, 768),
            "1024x1024": (1024, 1024), "1280x720": (1280, 720), "1920x1080": (1920, 1080),
        }
        
        if force_size == "Disabled":
            return src_w, src_h
        elif force_size == "Custom Width/Height":
            return custom_width, custom_height
        else:
            return size_map.get(force_size, (src_w, src_h))
    
    @classmethod
    def IS_CHANGED(cls, video, force_rate, force_size, custom_width, custom_height,
                   start_frame, end_frame, frame_load_cap, select_every_nth, **kwargs):
        video_path = os.path.join(folder_paths.get_input_directory(), video)
        if os.path.exists(video_path):
            m = hashlib.sha256()
            with open(video_path, 'rb') as f:
                m.update(f.read())
            return m.digest().hex()
        return float("NaN")


# ============================================================
# RK VIDEO SAVER - Save video from image batch
# ============================================================

class RK_VideoSaver:
    """Save a batch of images as a video file."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "rk_video"}),
                "format": (["mp4", "avi", "mov", "webm"], {"default": "mp4"}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "step": 0.1}),
                "codec": (["mp4v", "avc1", "XVID", "MJPG"], {"default": "mp4v"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1, "tooltip": "Video quality (CRF-like, higher=better)"}),
            },
        }
    
    RETURN_TYPES = ("STRING", "VIDEO")
    RETURN_NAMES = ("filepath", "video")
    OUTPUT_NODE = True
    CATEGORY = "rk_pix/video"
    FUNCTION = "save_video"
    
    def save_video(self, images, filename_prefix, format, fps, codec, quality):
        if not CV2_AVAILABLE:
            raise Exception("[RK_VideoSaver] OpenCV required. Install: pip install opencv-python")
        
        output_dir = folder_paths.get_output_directory()
        
        # Ensure unique filename
        counter = 1
        while True:
            filename = f"{filename_prefix}_{counter:04d}.{format}"
            filepath = os.path.join(output_dir, filename)
            if not os.path.exists(filepath):
                break
            counter += 1
        
        # Get dimensions
        b, h, w, c = images.shape
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
        
        if not writer.isOpened():
            # Fallback codec
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
        
        # Write frames
        for i in range(b):
            frame = images[i].cpu().numpy()
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        
        writer.release()
        
        info = f"Saved: {filename} | {b} frames | {w}x{h} | {fps}fps"
        print(f"[RK_VideoSaver] {info}")
        
        # Return video tensor for chaining
        return (filepath, images)


# ============================================================
# RK VIDEO COMBINE - Combine image batch into video format
# ============================================================

class RK_VideoCombine:
    """Convert image batch to VIDEO type for video-specific nodes."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "step": 0.1}),
            },
        }
    
    RETURN_TYPES = ("VIDEO", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("video", "frame_count", "fps", "info")
    CATEGORY = "rk_pix/video"
    FUNCTION = "combine"
    
    def combine(self, images, fps):
        frame_count = images.shape[0]
        info = f"Video: {frame_count} frames @ {fps}fps | Duration: {frame_count/fps:.2f}s"
        return (images, frame_count, fps, info)


# ============================================================
# RK VIDEO SPLIT - Split video into individual frames
# ============================================================

class RK_VideoSplit:
    """Split video into separate IMAGE outputs (up to 4 frames)."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("frame_1", "frame_2", "frame_3", "frame_4", "total_frames")
    CATEGORY = "rk_pix/video"
    FUNCTION = "split"
    
    def split(self, video, start_index):
        total = video.shape[0]
        frames = []
        
        for i in range(4):
            idx = start_index + i
            if idx < total:
                frames.append(video[idx:idx+1])
            else:
                # Return empty/black frame
                frames.append(torch.zeros((1, video.shape[1], video.shape[2], video.shape[3])))
        
        return (*frames, total)


# ============================================================
# RK VIDEO INFO - Get video metadata
# ============================================================

class RK_VideoInfo:
    """Display information about a video tensor."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            },
        }
    
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("batch", "height", "width", "channels", "info")
    OUTPUT_NODE = True
    CATEGORY = "rk_pix/video"
    FUNCTION = "get_info"
    
    def get_info(self, video):
        b, h, w, c = video.shape
        info = f"Shape: [{b}, {h}, {w}, {c}] | Frames: {b} | Resolution: {w}x{h} | Channels: {c}"
        return (b, h, w, c, info)


# ============================================================
# RK VIDEO FRAME EXTRACTOR - Extract specific frame as image
# ============================================================

class RK_VideoFrameExtractor:
    """Extract a single frame from video as IMAGE."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("frame", "total_frames", "info")
    CATEGORY = "rk_pix/video"
    FUNCTION = "extract"
    
    def extract(self, video, frame_index):
        total = video.shape[0]
        idx = min(frame_index, total - 1)
        frame = video[idx:idx+1]
        info = f"Extracted frame {idx + 1} / {total}"
        return (frame, total, info)


# ============================================================
# RK VIDEO SPEED - Adjust playback speed
# ============================================================

class RK_VideoSpeed:
    """Change video speed by duplicating or skipping frames."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "1.0=normal, 2.0=2x faster, 0.5=half speed"}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "step": 0.1}),
            },
        }
    
    RETURN_TYPES = ("VIDEO", "FLOAT", "STRING")
    RETURN_NAMES = ("video", "new_fps", "info")
    CATEGORY = "rk_pix/video"
    FUNCTION = "adjust_speed"
    
    def adjust_speed(self, video, speed, fps):
        total_frames = video.shape[0]
        
        if speed == 1.0:
            return (video, fps, f"Speed: 1.0x | {total_frames} frames")
        
        # Calculate new frame count
        new_count = max(1, int(total_frames / speed))
        
        # Sample frames
        indices = torch.linspace(0, total_frames - 1, new_count).long()
        new_video = video[indices]
        
        new_fps = fps * speed
        info = f"Speed: {speed}x | {total_frames} → {new_count} frames | FPS: {new_fps:.1f}"
        
        return (new_video, new_fps, info)


# ============================================================
# RK VIDEO REVERSE - Reverse video playback
# ============================================================

class RK_VideoReverse:
    """Reverse the order of frames in a video."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            },
        }
    
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("reversed_video", "info")
    CATEGORY = "rk_pix/video"
    FUNCTION = "reverse"
    
    def reverse(self, video):
        reversed_video = torch.flip(video, dims=[0])
        info = f"Reversed: {video.shape[0]} frames"
        return (reversed_video, info)


# ============================================================
# RK VIDEO LOOP - Loop video N times
# ============================================================

class RK_VideoLoop:
    """Loop a video multiple times."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "loop_count": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("VIDEO", "INT", "STRING")
    RETURN_NAMES = ("looped_video", "total_frames", "info")
    CATEGORY = "rk_pix/video"
    FUNCTION = "loop"
    
    def loop(self, video, loop_count):
        looped = video.repeat(loop_count, 1, 1, 1)
        total = looped.shape[0]
        info = f"Looped: {video.shape[0]} x {loop_count} = {total} frames"
        return (looped, total, info)


# ============================================================
# RK VIDEO FRAME PLAYER - Incremental frame-by-frame playback
# ============================================================

class RK_VideoFramePlayer:
    """
    Frame-by-frame incremental player for videos and images.
    
    Each time you run the workflow, it outputs the NEXT frame.
    Works with both VIDEO input and IMAGE batch input.
    
    Features:
    - Auto-advance: next frame on each run
    - Loop option: go back to start after last frame
    - Step size: skip N frames per run
    - Works with both VIDEO and IMAGE inputs
    - Outputs single IMAGE frame + VIDEO frame
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("VIDEO",),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1, "tooltip": "Frame to start from (0 = first frame)"}),
                "step": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "Frames to advance per run"}),
                "loop": ("BOOLEAN", {"default": True, "tooltip": "Loop back to first frame after last frame"}),
                "reset": ("BOOLEAN", {"default": False, "tooltip": "Reset to start frame on next run"}),
            },
            "optional": {
                "image_source": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("IMAGE", "VIDEO", "INT", "INT", "STRING")
    RETURN_NAMES = ("frame_image", "frame_video", "current_frame", "total_frames", "info")
    CATEGORY = "rk_pix/video"
    FUNCTION = "play_frame"
    
    # Class-level state: unique_id -> {current_index, start_index}
    _player_states = {}
    
    def play_frame(self, source, start_frame, step, loop, reset, image_source=None, unique_id=None):
        
        # Use VIDEO source if available, otherwise use IMAGE source
        if source is not None and source.shape[0] > 0:
            frames = source
        elif image_source is not None and image_source.shape[0] > 0:
            frames = image_source
        else:
            raise Exception("[RK_VideoFramePlayer] No valid input provided. Connect VIDEO or IMAGE.")
        
        total_frames = frames.shape[0]
        state_key = unique_id if unique_id else "default"
        
        # Clamp start_frame to valid range
        start_frame = min(start_frame, total_frames - 1)
        
        # Handle reset or initialize
        if reset:
            self._player_states[state_key] = {"current": start_frame, "start": start_frame}
            current = start_frame
        else:
            # Get or initialize state
            if state_key not in self._player_states:
                self._player_states[state_key] = {"current": start_frame, "start": start_frame}
            
            state = self._player_states[state_key]
            
            # If start_frame changed, reset to new start
            if state.get("start") != start_frame:
                state["current"] = start_frame
                state["start"] = start_frame
            
            current = state["current"]
            
            # Ensure valid
            current = current % total_frames
        
        # Extract current frame
        frame = frames[current:current+1]
        
        # Calculate next frame position
        next_frame = current + step
        
        if next_frame >= total_frames:
            if loop:
                next_frame = start_frame + (next_frame - total_frames) % (total_frames - start_frame) if start_frame > 0 else next_frame % total_frames
                if next_frame >= total_frames:
                    next_frame = start_frame
            else:
                next_frame = total_frames - 1
        
        # Save state for next run
        self._player_states[state_key]["current"] = next_frame
        
        # Build info with current frame number (1-based for display)
        frame_num = current + 1
        next_num = next_frame + 1
        start_num = start_frame + 1
        
        if loop and next_frame <= current and total_frames > 1 and next_frame == start_frame:
            info = f"▶ Frame {frame_num}/{total_frames} (Start: {start_num}) | Next: {next_num} (LOOPING)"
        else:
            info = f"▶ Frame {frame_num}/{total_frames} (Start: {start_num}) | Next: {next_num}"
        
        # Output as both IMAGE and VIDEO types
        frame_image = frame.clone()
        frame_video = frame.clone()
        
        return (frame_image, frame_video, current, total_frames, info)
    
    @classmethod
    def IS_CHANGED(cls, source, start_frame, step, loop, reset, image_source=None, unique_id=None):
        return float("NaN")


# ============================================================
# RK IMAGE FRAME PLAYER - Standalone image sequence player
# ============================================================

class RK_ImageFramePlayer:
    """
    Frame-by-frame player specifically for IMAGE batches.
    
    Simpler version that only accepts IMAGE input.
    Each run outputs the next frame as IMAGE.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1, "tooltip": "Frame to start from (0 = first frame)"}),
                "step": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "loop": ("BOOLEAN", {"default": True}),
                "reset": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("frame", "current_frame", "total_frames", "info")
    CATEGORY = "rk_pix/video"
    FUNCTION = "play"
    
    _player_states = {}
    
    def play(self, images, start_frame, step, loop, reset, unique_id=None):
        total = images.shape[0]
        state_key = unique_id if unique_id else "default"
        
        # Clamp start_frame to valid range
        start_frame = min(start_frame, total - 1)
        
        if reset:
            self._player_states[state_key] = {"current": start_frame, "start": start_frame}
            current = start_frame
        else:
            if state_key not in self._player_states:
                self._player_states[state_key] = {"current": start_frame, "start": start_frame}
            
            state = self._player_states[state_key]
            
            # If start_frame changed, reset to new start
            if state.get("start") != start_frame:
                state["current"] = start_frame
                state["start"] = start_frame
            
            current = state["current"] % total
        
        frame = images[current:current+1]
        
        next_frame = current + step
        if next_frame >= total:
            if loop:
                next_frame = start_frame + (next_frame - total) % (total - start_frame) if start_frame > 0 else next_frame % total
                if next_frame >= total:
                    next_frame = start_frame
            else:
                next_frame = total - 1
        
        self._player_states[state_key]["current"] = next_frame
        
        info = f"▶ Frame {current + 1}/{total} (Start: {start_frame + 1}) | Next: {next_frame + 1}"
        return (frame, current, total, info)
    
    @classmethod
    def IS_CHANGED(cls, images, start_frame, step, loop, reset, unique_id=None):
        return float("NaN")


# ============================================================
# REGISTRATION
# ============================================================

WEB_DIRECTORY = "./js"

VIDEO_CLASS_MAPPINGS = {
    "RK_VideoLoader": RK_VideoLoader,
    "RK_VideoSaver": RK_VideoSaver,
    "RK_VideoCombine": RK_VideoCombine,
    "RK_VideoSplit": RK_VideoSplit,
    "RK_VideoInfo": RK_VideoInfo,
    "RK_VideoFrameExtractor": RK_VideoFrameExtractor,
    "RK_VideoSpeed": RK_VideoSpeed,
    "RK_VideoReverse": RK_VideoReverse,
    "RK_VideoLoop": RK_VideoLoop,
    "RK_VideoFramePlayer": RK_VideoFramePlayer,
    "RK_ImageFramePlayer": RK_ImageFramePlayer,
}

VIDEO_NAME_MAPPINGS = {
    "RK_VideoLoader": "Video Loader (RK)",
    "RK_VideoSaver": "Video Saver (RK)",
    "RK_VideoCombine": "Video Combine (RK)",
    "RK_VideoSplit": "Video Split (RK)",
    "RK_VideoInfo": "Video Info (RK)",
    "RK_VideoFrameExtractor": "Video Frame Extractor (RK)",
    "RK_VideoSpeed": "Video Speed (RK)",
    "RK_VideoReverse": "Video Reverse (RK)",
    "RK_VideoLoop": "Video Loop (RK)",
    "RK_VideoFramePlayer": "Video Frame Player (RK)",
    "RK_ImageFramePlayer": "Image Frame Player (RK)",
}
