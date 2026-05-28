import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * RK Video Nodes - Frontend Extension
 * Fixed for cross-system compatibility
 */

const RK_VIDEO_EXT_NAME = "RK_Pix.VideoNodesUI";

app.registerExtension({
    name: RK_VIDEO_EXT_NAME,

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RK_VideoLoader") {
            this._setupVideoLoaderNode(nodeType, nodeData);
        }
        if (nodeData.name === "RK_VideoSaver") {
            this._setupVideoSaverNode(nodeType, nodeData);
        }
        if (nodeData.name === "RK_VideoFramePlayer") {
            this._setupFramePlayerNode(nodeType, nodeData);
        }
        if (nodeData.name === "RK_ImageFramePlayer") {
            this._setupImagePlayerNode(nodeType, nodeData);
        }
    },

    // ============================================================
    // RK_VideoLoader Node UI
    // ============================================================
    _setupVideoLoaderNode(nodeType, nodeData) {
        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const node = this;
            
            // Find video widget - handle different widget structures
            let videoWidget = null;
            for (let i = 0; i < node.widgets.length; i++) {
                if (node.widgets[i].name === "video") {
                    videoWidget = node.widgets[i];
                    break;
                }
            }
            
            if (!videoWidget) {
                console.warn("[RK_VideoNodesUI] Video widget not found on node", node);
                return;
            }

            // ---- UPLOAD BUTTON ----
            try {
                const uploadBtn = node.addWidget("button", "upload_video", "📁 Upload", () => {
                    const input = document.createElement("input");
                    input.type = "file";
                    input.accept = "video/*,image/*,.mp4,.avi,.mov,.mkv,.wmv,.flv,.webm,.m4v,.png,.jpg,.jpeg,.webp";
                    input.style.display = "none";
                    
                    input.onchange = async (e) => {
                        const file = e.target.files[0];
                        if (!file) return;
                        await RKVideoUploader.uploadFile(file, node, videoWidget);
                    };
                    
                    document.body.appendChild(input);
                    input.click();
                    setTimeout(() => document.body.removeChild(input), 1000);
                });
                if (uploadBtn) uploadBtn.serialize = false;
            } catch (err) {
                console.warn("[RK_VideoNodesUI] Could not add upload button:", err);
            }

            // ---- PREVIEW CONTAINER ----
            const previewContainer = document.createElement("div");
            previewContainer.className = "rk-video-preview";
            previewContainer.style.cssText = `
                width: 280px !important;
                min-height: 60px;
                background: #1a1a2e;
                border-radius: 8px;
                padding: 8px;
                margin-top: 6px;
                display: none;
                box-sizing: border-box;
            `;

            // Video element
            const previewMedia = document.createElement("video");
            previewMedia.style.cssText = `
                width: 100% !important;
                max-width: 264px;
                border-radius: 4px;
                display: block;
                background: #000;
                min-height: 40px;
            `;
            previewMedia.controls = true;
            previewMedia.preload = "metadata";
            previewMedia.muted = true;
            previewContainer.appendChild(previewMedia);

            // Image element (fallback)
            const previewImage = document.createElement("img");
            previewImage.style.cssText = `
                width: 100% !important;
                max-width: 264px;
                border-radius: 4px;
                display: none;
                background: #000;
                min-height: 40px;
            `;
            previewImage.alt = "Preview";
            previewContainer.appendChild(previewImage);

            // Info display
            const infoDisplay = document.createElement("div");
            infoDisplay.style.cssText = `
                color: #aaa;
                font-size: 11px;
                margin-top: 6px;
                font-family: monospace;
                line-height: 1.4;
                word-break: break-all;
            `;
            previewContainer.appendChild(infoDisplay);

            // Range display
            const rangeDisplay = document.createElement("div");
            rangeDisplay.style.cssText = `
                color: #e94560;
                font-size: 11px;
                margin-top: 4px;
                font-family: monospace;
                background: rgba(233, 69, 96, 0.1);
                padding: 4px 8px;
                border-radius: 4px;
                display: none;
            `;
            previewContainer.appendChild(rangeDisplay);

            // Frame counter
            const frameCounter = document.createElement("div");
            frameCounter.style.cssText = `
                color: #e94560;
                font-size: 12px;
                margin-top: 4px;
                text-align: center;
                font-weight: bold;
            `;
            previewContainer.appendChild(frameCounter);

            // Output indicator
            const outputIndicator = document.createElement("div");
            outputIndicator.style.cssText = `
                color: #4ecca3;
                font-size: 10px;
                margin-top: 4px;
                text-align: center;
                font-family: monospace;
                background: rgba(78, 204, 163, 0.1);
                padding: 2px 6px;
                border-radius: 3px;
            `;
            outputIndicator.textContent = "Outputs: IMAGE + VIDEO";
            previewContainer.appendChild(outputIndicator);

            // Add DOM widget
            try {
                const previewWidget = node.addDOMWidget("rk_preview", "custom", previewContainer);
                if (previewWidget) previewWidget.serialize = false;
            } catch (err) {
                console.warn("[RK_VideoNodesUI] Could not add preview widget:", err);
            }

            // ---- DRAG & DROP ----
            const dropOverlay = document.createElement("div");
            dropOverlay.style.cssText = `
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(233, 69, 96, 0.2);
                border: 3px dashed #e94560;
                border-radius: 8px;
                display: none;
                align-items: center;
                justify-content: center;
                z-index: 1000;
                pointer-events: none;
                font-size: 16px;
                font-weight: bold;
                color: #e94560;
            `;
            dropOverlay.textContent = "📥 Drop Video/Image";

            try {
                const dropWidget = node.addDOMWidget("rk_drop", "custom", dropOverlay);
                if (dropWidget) dropWidget.serialize = false;
            } catch (err) {
                console.warn("[RK_VideoNodesUI] Could not add drop widget:", err);
            }

            // Drag handlers
            const originalOnDragOver = node.onDragOver;
            node.onDragOver = function (e) {
                if (e.dataTransfer && e.dataTransfer.types) {
                    const types = Array.from(e.dataTransfer.types);
                    if (types.includes("Files") || types.includes("application/x-moz-file")) {
                        e.preventDefault();
                        if (dropOverlay) dropOverlay.style.display = "flex";
                    }
                }
                if (originalOnDragOver) originalOnDragOver.call(this, e);
            };

            const originalOnDragLeave = node.onDragLeave;
            node.onDragLeave = function (e) {
                if (dropOverlay) dropOverlay.style.display = "none";
                if (originalOnDragLeave) originalOnDragLeave.call(this, e);
            };

            const originalOnDrop = node.onDrop;
            node.onDrop = async function (e) {
                if (dropOverlay) dropOverlay.style.display = "none";
                
                if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                    const file = e.dataTransfer.files[0];
                    const isVideo = file.type.startsWith("video/") || 
                        /\.(mp4|avi|mov|mkv|wmv|flv|webm|m4v)$/i.test(file.name);
                    const isImage = file.type.startsWith("image/") || 
                        /\.(png|jpg|jpeg|webp|bmp|tiff)$/i.test(file.name);

                    if (isVideo || isImage) {
                        e.preventDefault();
                        e.stopPropagation();
                        await RKVideoUploader.uploadFile(file, node, videoWidget);
                        return;
                    }
                }
                if (originalOnDrop) originalOnDrop.call(this, e);
            };

            // Watch Start/End frame widgets
            let startFrameWidget = null;
            let endFrameWidget = null;
            for (let w of node.widgets) {
                if (w.name === "start_frame") startFrameWidget = w;
                if (w.name === "end_frame") endFrameWidget = w;
            }
            
            const updateRangeDisplay = () => {
                if (!rangeDisplay) return;
                const start = startFrameWidget ? startFrameWidget.value : 0;
                const end = endFrameWidget ? endFrameWidget.value : -1;
                if (start > 0 || end >= 0) {
                    rangeDisplay.style.display = "block";
                    rangeDisplay.textContent = `Range: ${start} → ${end >= 0 ? end : 'end'}`;
                } else {
                    rangeDisplay.style.display = "none";
                }
            };
            
            if (startFrameWidget && startFrameWidget.callback) {
                const origCb = startFrameWidget.callback;
                startFrameWidget.callback = function(v) { origCb(v); updateRangeDisplay(); };
            }
            if (endFrameWidget && endFrameWidget.callback) {
                const origCb = endFrameWidget.callback;
                endFrameWidget.callback = function(v) { origCb(v); updateRangeDisplay(); };
            }

            // Update preview on video change
            const originalCallback = videoWidget.callback;
            videoWidget.callback = async function (value) {
                if (originalCallback) originalCallback.apply(this, arguments);
                try {
                    await RKVideoPreview.updatePreview(
                        node, value, previewMedia, previewImage, 
                        infoDisplay, frameCounter, previewContainer,
                        rangeDisplay, startFrameWidget, endFrameWidget
                    );
                } catch (err) {
                    console.warn("[RK_VideoNodesUI] Preview update failed:", err);
                }
            };

            // Initial preview
            if (videoWidget.value) {
                setTimeout(() => {
                    try {
                        RKVideoPreview.updatePreview(
                            node, videoWidget.value, previewMedia, previewImage,
                            infoDisplay, frameCounter, previewContainer,
                            rangeDisplay, startFrameWidget, endFrameWidget
                        );
                    } catch (err) {
                        console.warn("[RK_VideoNodesUI] Initial preview failed:", err);
                    }
                }, 500);
            }
        };
    },

    // ============================================================
    // RK_VideoFramePlayer Node UI
    // ============================================================
    _setupFramePlayerNode(nodeType, nodeData) {
        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const node = this;

            const frameDisplay = document.createElement("div");
            frameDisplay.style.cssText = `
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                border-radius: 8px;
                padding: 12px;
                color: #e94560;
                font-size: 16px;
                font-weight: bold;
                text-align: center;
                font-family: monospace;
                margin-top: 4px;
                border: 1px solid rgba(233, 69, 96, 0.3);
            `;
            frameDisplay.innerHTML = `<div style="font-size:18px;margin-bottom:4px;">▶ Frame <span style="color:#fff;">-</span> / -</div>`;

            try {
                const displayWidget = node.addDOMWidget("rk_frame_counter", "custom", frameDisplay);
                if (displayWidget) displayWidget.serialize = false;
            } catch (err) {}

            node.rkFrameDisplay = frameDisplay;

            const progressContainer = document.createElement("div");
            progressContainer.style.cssText = `
                width: 100%;
                height: 8px;
                background: #333;
                border-radius: 4px;
                margin-top: 8px;
                overflow: hidden;
            `;

            const progressBar = document.createElement("div");
            progressBar.style.cssText = `
                height: 100%;
                background: linear-gradient(90deg, #e94560, #ff6b6b);
                width: 0%;
                transition: width 0.3s ease;
            `;
            progressContainer.appendChild(progressBar);

            try {
                const progressWidget = node.addDOMWidget("rk_progress", "custom", progressContainer);
                if (progressWidget) progressWidget.serialize = false;
            } catch (err) {}

            node.rkProgressBar = progressBar;

            const statusDisplay = document.createElement("div");
            statusDisplay.style.cssText = `
                color: #4ecca3;
                font-size: 10px;
                margin-top: 6px;
                text-align: center;
                font-family: monospace;
            `;
            statusDisplay.textContent = "Queue prompt to advance frame";

            try {
                const statusWidget = node.addDOMWidget("rk_status", "custom", statusDisplay);
                if (statusWidget) statusWidget.serialize = false;
            } catch (err) {}

            node.rkStatusDisplay = statusDisplay;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            if (this.rkFrameDisplay && message) {
                try {
                    const current = message.current_frame !== undefined ? message.current_frame[0] : 0;
                    const total = message.total_frames !== undefined ? message.total_frames[0] : 0;
                    const info = message.info ? message.info[0] : "";
                    
                    let startInfo = "";
                    const startMatch = info.match(/Start:\s*(\d+)/);
                    if (startMatch) {
                        startInfo = ` <span style="color:#888;font-size:12px;">(from #${startMatch[1]})</span>`;
                    }
                    
                    this.rkFrameDisplay.innerHTML = `
                        <div style="font-size:18px;margin-bottom:4px;">▶ Frame <span style="color:#fff;">${current + 1}</span> / ${total}${startInfo}</div>
                        <div style="font-size:11px;color:#aaa;font-weight:normal;">${info.replace(/▶ Frame \d+\/\d+.*?\| /, '')}</div>
                    `;
                    
                    if (this.rkProgressBar && total > 0) {
                        const pct = ((current + 1) / total) * 100;
                        this.rkProgressBar.style.width = `${pct}%`;
                    }
                    
                    if (this.rkStatusDisplay) {
                        this.rkStatusDisplay.textContent = "Next run will advance frame";
                    }
                } catch (err) {
                    console.warn("[RK_VideoNodesUI] Frame display update failed:", err);
                }
            }
        };
    },

    // ============================================================
    // RK_ImageFramePlayer Node UI
    // ============================================================
    _setupImagePlayerNode(nodeType, nodeData) {
        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const node = this;

            const frameDisplay = document.createElement("div");
            frameDisplay.style.cssText = `
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                border-radius: 8px;
                padding: 12px;
                color: #e94560;
                font-size: 16px;
                font-weight: bold;
                text-align: center;
                font-family: monospace;
                margin-top: 4px;
                border: 1px solid rgba(233, 69, 96, 0.3);
            `;
            frameDisplay.innerHTML = `<div style="font-size:18px;margin-bottom:4px;">▶ Frame <span style="color:#fff;">-</span> / -</div>`;

            try {
                const displayWidget = node.addDOMWidget("rk_frame_counter", "custom", frameDisplay);
                if (displayWidget) displayWidget.serialize = false;
            } catch (err) {}

            node.rkFrameDisplay = frameDisplay;

            const progressContainer = document.createElement("div");
            progressContainer.style.cssText = `
                width: 100%;
                height: 8px;
                background: #333;
                border-radius: 4px;
                margin-top: 8px;
                overflow: hidden;
            `;

            const progressBar = document.createElement("div");
            progressBar.style.cssText = `
                height: 100%;
                background: linear-gradient(90deg, #e94560, #ff6b6b);
                width: 0%;
                transition: width 0.3s ease;
            `;
            progressContainer.appendChild(progressBar);

            try {
                const progressWidget = node.addDOMWidget("rk_progress", "custom", progressContainer);
                if (progressWidget) progressWidget.serialize = false;
            } catch (err) {}

            node.rkProgressBar = progressBar;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            if (this.rkFrameDisplay && message) {
                try {
                    const current = message.current_frame !== undefined ? message.current_frame[0] : 0;
                    const total = message.total_frames !== undefined ? message.total_frames[0] : 0;
                    const info = message.info ? message.info[0] : "";
                    
                    let startInfo = "";
                    const startMatch = info.match(/Start:\s*(\d+)/);
                    if (startMatch) {
                        startInfo = ` <span style="color:#888;font-size:12px;">(from #${startMatch[1]})</span>`;
                    }
                    
                    this.rkFrameDisplay.innerHTML = `
                        <div style="font-size:18px;margin-bottom:4px;">▶ Frame <span style="color:#fff;">${current + 1}</span> / ${total}${startInfo}</div>
                        <div style="font-size:11px;color:#aaa;font-weight:normal;">${info.replace(/▶ Frame \d+\/\d+.*?\| /, '')}</div>
                    `;
                    
                    if (this.rkProgressBar && total > 0) {
                        const pct = ((current + 1) / total) * 100;
                        this.rkProgressBar.style.width = `${pct}%`;
                    }
                } catch (err) {
                    console.warn("[RK_VideoNodesUI] Frame display update failed:", err);
                }
            }
        };
    },

    // ============================================================
    // RK_VideoSaver Node UI
    // ============================================================
    _setupVideoSaverNode(nodeType, nodeData) {
        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const node = this;

            const statusDisplay = document.createElement("div");
            statusDisplay.style.cssText = `
                background: #1a1a2e;
                border-radius: 6px;
                padding: 8px 12px;
                color: #4ecca3;
                font-size: 11px;
                font-family: monospace;
                margin-top: 4px;
                text-align: center;
            `;
            statusDisplay.textContent = "Ready to save video";

            try {
                const statusWidget = node.addDOMWidget("rk_status", "custom", statusDisplay);
                if (statusWidget) statusWidget.serialize = false;
            } catch (err) {}

            node.rkStatusDisplay = statusDisplay;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            if (this.rkStatusDisplay && message) {
                try {
                    const filepath = message.filepath ? message.filepath[0] : "";
                    this.rkStatusDisplay.textContent = filepath ? `Saved: ${filepath}` : "Video saved";
                } catch (err) {}
            }
        };
    },
});

// ============================================================
// Helper Classes
// ============================================================

class RKVideoUploader {
    static async uploadFile(file, node, videoWidget) {
        const formData = new FormData();
        formData.append("image", file);
        formData.append("type", "input");
        formData.append("subfolder", "");

        try {
            const uploadBtn = node.widgets.find(w => w.name === "upload_video");
            if (uploadBtn) uploadBtn.value = "Uploading...";

            const resp = await api.fetchApi("/upload/image", {
                method: "POST",
                body: formData,
            });

            if (resp.status === 200) {
                const data = await resp.json();
                const filename = data.name;

                videoWidget.value = filename;
                if (videoWidget.callback) {
                    videoWidget.callback(filename);
                }

                app.graph.setDirtyCanvas(true);

                if (uploadBtn) uploadBtn.value = "Uploaded!";
                setTimeout(() => { 
                    if (uploadBtn) uploadBtn.value = "Upload"; 
                }, 2000);
            } else {
                throw new Error(`Upload failed: ${resp.status}`);
            }
        } catch (err) {
            console.error("[RK_VideoNodesUI] Upload error:", err);
            alert("Upload failed: " + err.message);
            
            const uploadBtn = node.widgets.find(w => w.name === "upload_video");
            if (uploadBtn) uploadBtn.value = "Upload Failed";
        }
    }
}

class RKVideoPreview {
    static async updatePreview(
        node, filename, videoEl, imageEl, infoEl, counterEl, 
        container, rangeEl, startWidget, endWidget
    ) {
        if (!filename || !container) {
            if (container) container.style.display = "none";
            return;
        }

        const isVideo = /\.(mp4|avi|mov|mkv|wmv|flv|webm|m4v)$/i.test(filename);
        const isImage = /\.(png|jpg|jpeg|webp|bmp|tiff|tif|gif)$/i.test(filename);

        if (!isVideo && !isImage) {
            container.style.display = "none";
            return;
        }

        const url = `/view?filename=${encodeURIComponent(filename)}&type=input`;

        if (isVideo) {
            videoEl.style.display = "block";
            imageEl.style.display = "none";
            videoEl.src = url;

            videoEl.onloadedmetadata = () => {
                try {
                    const duration = videoEl.duration || 0;
                    const fps = 30;
                    const estimatedFrames = Math.floor(duration * fps);
                    
                    infoEl.innerHTML = `
                        ${filename}<br>
                        ${duration.toFixed(2)}s | ${videoEl.videoWidth}x${videoEl.videoHeight}<br>
                        ~${estimatedFrames} frames
                    `;
                    counterEl.textContent = `${estimatedFrames} frames`;
                    
                    if (rangeEl && startWidget && endWidget) {
                        const start = startWidget.value || 0;
                        const end = endWidget.value >= 0 ? endWidget.value : estimatedFrames;
                        if (start > 0 || endWidget.value >= 0) {
                            rangeEl.style.display = "block";
                            rangeEl.textContent = `Range: ${start} -> ${end >= 0 ? end : 'end'}`;
                        } else {
                            rangeEl.style.display = "none";
                        }
                    }
                } catch (err) {
                    console.warn("[RK_VideoNodesUI] Video metadata error:", err);
                }
            };

            videoEl.onerror = () => {
                infoEl.innerHTML = `Error loading: ${filename}`;
                counterEl.textContent = "";
            };

        } else if (isImage) {
            videoEl.style.display = "none";
            imageEl.style.display = "block";
            imageEl.src = url;

            imageEl.onload = () => {
                infoEl.innerHTML = `
                    ${filename}<br>
                    ${imageEl.naturalWidth}x${imageEl.naturalHeight}
                `;
                counterEl.textContent = "Single Image";
                if (rangeEl) rangeEl.style.display = "none";
            };

            imageEl.onerror = () => {
                infoEl.innerHTML = `Error loading: ${filename}`;
            };
        }

        container.style.display = "block";
        
        // Force canvas redraw
        if (app && app.graph) {
            app.graph.setDirtyCanvas(true, true);
        }
    }
}
