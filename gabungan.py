# gabungan_upgraded.py
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import numpy as np
import cv2
import io
import os
import math
import zipfile
from typing import Tuple

# ------------------------------------------------------------------
# 1️⃣  BACKGROUND-REMOVAL MODEL (unchanged)
# ------------------------------------------------------------------
try:
    from model import U2NET, U2NETP
except ImportError:
    st.error("❌ model.py not found – place U2NET classes in same folder")
    st.stop()

class EnhancedU2NetPredictor:
    def __init__(self, model_path: str, model_type: str = "U2NET"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.model = self.load_model(model_path)

        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.hr_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path: str):
        try:
            net = U2NET(3, 1) if self.model_type == "U2NET" else U2NETP(3, 1)
            net.load_state_dict(torch.load(model_path,
                                           map_location=None if torch.cuda.is_available() else 'cpu'))
            net.to(self.device).eval()
            return net
        except Exception as e:
            st.error(f"Model load error: {e}")
            return None

    def enhance_for_objects(self, img: Image.Image) -> Image.Image:
        img = ImageEnhance.Contrast(img).enhance(1.2)
        img = ImageEnhance.Sharpness(img).enhance(1.1)
        img = ImageEnhance.Color(img).enhance(1.05)
        return img

    def predict_enhanced(self, img: Image.Image, enhance=True, high_res=False) -> np.ndarray:
        if self.model is None:
            return None
        img = self.enhance_for_objects(img) if enhance else img
        tensor = (self.hr_transform if high_res else self.transform)(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(tensor)[0]
        pred_np = pred.cpu().numpy().squeeze()
        return cv2.resize(pred_np, img.size, interpolation=cv2.INTER_CUBIC)

    def post_process_mask(self, pred, thresh=0.5, smooth=True, denoise=True) -> np.ndarray:
        mask = (pred > thresh).astype(np.uint8) * 255
        if denoise:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if smooth:
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            mask = (mask > 127).astype(np.uint8) * 255
        return mask

    def remove_bg(self, img, mask, feather=True, smooth=2) -> Image.Image:
        if feather and smooth:
            mask = cv2.GaussianBlur(mask, (smooth * 2 + 1, smooth * 2 + 1), 0)
        mask_norm = mask.astype(np.float32) / 255.0
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            masked = img_np * np.stack([mask_norm] * 3, axis=2)
            alpha = (mask_norm * 255).astype(np.uint8)
            return Image.fromarray(np.dstack([masked.astype(np.uint8), alpha]), 'RGBA')
        return Image.fromarray((img_np * mask_norm).astype(np.uint8))

class ImageCropper:
    @staticmethod
    def crop(img: Image.Image, ar: str) -> Image.Image:
        w, h = img.size
        if ar == "1:1":
            s = min(w, h)
            l, t = (w - s) // 2, (h - s) // 2
            return img.crop((l, t, l + s, t + s))
        if ar == "3:4":
            new_w, new_h = (int(h * 3 / 4), h) if w / h > 3 / 4 else (w, int(w * 4 / 3))
            l, t = (w - new_w) // 2, (h - new_h) // 2
            return img.crop((l, t, l + new_w, t + new_h))
        if ar == "4:3":
            new_w, new_h = (int(h * 4 / 3), h) if w / h > 4 / 3 else (w, int(w * 3 / 4))
            l, t = (w - new_w) // 2, (h - new_h) // 2
            return img.crop((l, t, l + new_w, t + new_h))
        return img

# ------------------------------------------------------------------
# 2️⃣  ENHANCED STUDIO BACKGROUND GENERATOR
# ------------------------------------------------------------------
class StudioBackgroundGenerator:
    def __init__(self):
        self.gradient_presets = {
            "🤍 Professional White": [(255, 255, 255), (248, 248, 248)],
            "🩶 Soft Gray": [(240, 240, 240), (220, 220, 220)],
            "🤎 Warm Studio": [(255, 252, 248), (245, 240, 235)],
            "💙 Cool Studio": [(248, 250, 255), (240, 245, 250)],
            "🖤 Luxury Black": [(40, 40, 40), (20, 20, 20)],
            "💗 Fashion Pink": [(255, 245, 250), (250, 235, 245)],
            "💚 Mint Fresh": [(245, 255, 250), (235, 250, 245)],
            "🧡 Sunset Warm": [(255, 248, 240), (255, 235, 215)],
            "💎 Ocean Blue": [(240, 248, 255), (225, 240, 255)],
            "⚪ Pure White": [(255, 255, 255), (255, 255, 255)],
            "🌈 Rainbow Gradient": [(255, 182, 193), (135, 206, 235)],
            "🌅 Golden Hour": [(255, 215, 0), (255, 140, 0)],
            "🌸 Cherry Blossom": [(255, 182, 193), (255, 228, 225)],
            "🌊 Deep Ocean": [(25, 25, 112), (70, 130, 180)],
            "🍃 Nature Green": [(144, 238, 144), (34, 139, 34)]
        }
        
        self.texture_presets = {
            "None": None,
            "Paper": "paper",
            "Canvas": "canvas", 
            "Marble": "marble",
            "Wood": "wood",
            "Metal": "metal"
        }

    def create_gradient_background(self, size, colors, direction="vertical", curve="linear"):
        w, h = size
        img = Image.new("RGB", size)
        dr = ImageDraw.Draw(img)
        c1, c2 = colors
        
        if direction == "vertical":
            for y in range(h):
                ratio = self._apply_curve(y / h, curve)
                color = tuple(int(c1[i] * (1 - ratio) + c2[i] * ratio) for i in range(3))
                dr.line([(0, y), (w, y)], fill=color)
        elif direction == "horizontal":
            for x in range(w):
                ratio = self._apply_curve(x / w, curve)
                color = tuple(int(c1[i] * (1 - ratio) + c2[i] * ratio) for i in range(3))
                dr.line([(x, 0), (x, h)], fill=color)
        elif direction == "diagonal":
            for y in range(h):
                for x in range(w):
                    ratio = self._apply_curve((x + y) / (w + h), curve)
                    color = tuple(int(c1[i] * (1 - ratio) + c2[i] * ratio) for i in range(3))
                    img.putpixel((x, y), color)
        elif direction == "radial":
            cx, cy = w // 2, h // 2
            max_r = math.sqrt((w / 2) ** 2 + (h / 2) ** 2)
            for y in range(h):
                for x in range(w):
                    r = math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_r
                    ratio = self._apply_curve(min(1.0, r), curve)
                    color = tuple(int(c1[i] * (1 - ratio) + c2[i] * ratio) for i in range(3))
                    img.putpixel((x, y), color)
        elif direction == "diamond":
            cx, cy = w // 2, h // 2
            max_d = max(cx, cy)
            for y in range(h):
                for x in range(w):
                    d = (abs(x - cx) + abs(y - cy)) / max_d
                    ratio = self._apply_curve(min(1.0, d), curve)
                    color = tuple(int(c1[i] * (1 - ratio) + c2[i] * ratio) for i in range(3))
                    img.putpixel((x, y), color)
        
        return img

    def _apply_curve(self, t, curve):
        if curve == "ease_in":
            return t ** 2
        elif curve == "ease_out":
            return 1 - (1 - t) ** 2
        elif curve == "ease_in_out":
            return 0.5 * (1 - math.cos(math.pi * t))
        elif curve == "bounce":
            return abs(math.sin(t * math.pi * 2)) * t
        else:  # linear
            return t

    def add_texture_overlay(self, img, texture_type, intensity=0.1):
        if texture_type is None:
            return img
        
        w, h = img.size
        overlay = Image.new("L", (w, h))
        
        if texture_type == "paper":
            # Create paper-like texture
            for y in range(h):
                for x in range(w):
                    noise = (hash((x * 17 + y * 23) % 1000) % 50 - 25) * intensity
                    gray = 128 + noise
                    overlay.putpixel((x, y), int(max(0, min(255, gray))))
        
        # Convert to RGB and blend
        overlay_rgb = Image.new("RGB", (w, h))
        for y in range(h):
            for x in range(w):
                val = overlay.getpixel((x, y))
                overlay_rgb.putpixel((x, y), (val, val, val))
        
        return Image.blend(img, overlay_rgb, intensity)

    def create_studio_lighting_effect(self, img, light_pos="top", intensity=0.3):
        w, h = img.size
        lighting = Image.new("RGBA", (w, h), (255, 255, 255, 0))
        draw = ImageDraw.Draw(lighting)
        
        if light_pos == "top":
            # Soft top lighting
            for y in range(h // 3):
                alpha = int(intensity * 255 * (1 - y / (h // 3)))
                draw.rectangle([(0, y), (w, y + 1)], fill=(255, 255, 255, alpha))
        elif light_pos == "center":
            # Center spotlight
            cx, cy = w // 2, h // 2
            max_r = min(w, h) // 2
            for r in range(max_r):
                alpha = int(intensity * 255 * (1 - r / max_r))
                draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=(255, 255, 255, alpha))
        
        base = img.convert("RGBA")
        return Image.alpha_composite(base, lighting).convert("RGB")

    def create_drop_shadow(self, image: Image.Image, intensity: float = 0.3, blur_radius: int = 5, offset: tuple = (5, 5)) -> Image.Image:
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Create shadow from alpha channel
        alpha = image.split()[-1]
        shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        
        # Apply blur and intensity
        shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        shadow_alpha = shadow_alpha.point(lambda p: int(p * intensity))
        shadow.putalpha(shadow_alpha)
        
        return shadow

    def create_floor_reflection(self, image: Image.Image, intensity: float = 0.4, 
                           gradient_start: float = 0.8, fade_height: float = 0.6, 
                           bg_is_light: bool = True) -> Image.Image:
        """
        Create realistic floor reflection effect - OPTIMIZED FOR WHITE BACKGROUNDS
        """
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        
        # Get image dimensions
        width, height = image.size
        
        # Create flipped reflection
        reflection = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Calculate reflection height
        reflection_height = int(height * fade_height)
        
        # Crop reflection to desired height
        reflection = reflection.crop((0, 0, width, reflection_height))
        
        # For white backgrounds, we need to darken and increase contrast of reflection
        if bg_is_light:
            # Convert to array for easier manipulation
            reflection_array = np.array(reflection)
            
            # Darken the reflection for better visibility on white
            r, g, b, a = reflection_array[:,:,0], reflection_array[:,:,1], reflection_array[:,:,2], reflection_array[:,:,3]
            
            # Apply darkening only where alpha > 0 (subject pixels)
            mask = a > 0
            
            # Darken RGB values by 30-50% for better contrast on white
            darkening_factor = 0.6
            r[mask] = (r[mask] * darkening_factor).astype(np.uint8)
            g[mask] = (g[mask] * darkening_factor).astype(np.uint8) 
            b[mask] = (b[mask] * darkening_factor).astype(np.uint8)
            
            # Increase contrast slightly
            contrast_factor = 1.2
            r[mask] = np.clip(((r[mask] - 127) * contrast_factor + 127), 0, 255).astype(np.uint8)
            g[mask] = np.clip(((g[mask] - 127) * contrast_factor + 127), 0, 255).astype(np.uint8)
            b[mask] = np.clip(((b[mask] - 127) * contrast_factor + 127), 0, 255).astype(np.uint8)
            
            # Reconstruct image
            reflection_array[:,:,0] = r
            reflection_array[:,:,1] = g  
            reflection_array[:,:,2] = b
            
            reflection = Image.fromarray(reflection_array, 'RGBA')
        
        # Create gradient mask with higher initial intensity for white backgrounds
        base_intensity = intensity * 1.5 if bg_is_light else intensity
        
        gradient_mask = Image.new("L", (width, reflection_height), 0)
        
        # Draw gradient from top to bottom
        for y in range(reflection_height):
            progress = y / reflection_height
            
            if progress < gradient_start:
                # Start stronger for white backgrounds
                alpha_value = int(base_intensity * 255 * (1 - progress * 0.2))
            else:
                # Gentler fade
                remaining = (progress - gradient_start) / (1 - gradient_start) 
                alpha_value = int(base_intensity * 255 * (1 - gradient_start * 0.2) * (1 - remaining ** 1.2))
            
            # Ensure alpha is within bounds  
            alpha_value = max(0, min(255, alpha_value))
            
            # Fill entire row
            for x in range(width):
                gradient_mask.putpixel((x, y), alpha_value)
        
        # Apply gradient mask to reflection's alpha channel
        r, g, b, a = reflection.split()
        
        # Combine original alpha with gradient mask
        new_alpha = Image.new("L", (width, reflection_height), 0)
        
        for y in range(reflection_height):
            for x in range(width):
                original_alpha = a.getpixel((x, y))
                mask_alpha = gradient_mask.getpixel((x, y))
                
                # Multiply alphas
                combined_alpha = int((original_alpha * mask_alpha) / 255)
                new_alpha.putpixel((x, y), combined_alpha)
        
        # Create final reflection
        final_reflection = Image.merge("RGBA", (r, g, b, new_alpha))
        
        return final_reflection

    def composite_with_background(self, subject, background, position="center", scale=1.0, 
                            shadow=True, shadow_intensity=0.3, shadow_blur=5, shadow_offset=(5, 5),
                            lighting="none", lighting_intensity=0.2,
                            reflection=False, reflection_intensity=0.4, reflection_fade=0.6):
        
        # Ensure subject has alpha channel
        if subject.mode != "RGBA":
            subject = subject.convert("RGBA")
        
        # Scale subject if needed
        if scale != 1.0:
            new_size = (int(subject.width * scale), int(subject.height * scale))
            subject = subject.resize(new_size, Image.Resampling.LANCZOS)
        
        # Calculate position
        bg_w, bg_h = background.size
        sub_w, sub_h = subject.size
        
        if position == "center":
            x, y = (bg_w - sub_w) // 2, (bg_h - sub_h) // 2
        elif position == "bottom_center":
            x, y = (bg_w - sub_w) // 2, bg_h - sub_h - 20
        elif position == "top_center":
            x, y = (bg_w - sub_w) // 2, 20
        elif position == "left_center":
            x, y = 20, (bg_h - sub_h) // 2
        elif position == "right_center":
            x, y = bg_w - sub_w - 20, (bg_h - sub_h) // 2
        else:
            x, y = (bg_w - sub_w) // 2, (bg_h - sub_h) // 2
        
        # Create composite canvas
        canvas = background.convert("RGBA")
        
        # Add lighting effect to background if requested
        if lighting != "none":
            canvas = self.create_studio_lighting_effect(canvas.convert("RGB"), lighting, lighting_intensity).convert("RGBA")
        
        # Add floor reflection if enabled (render behind everything) - OPTIMIZED FOR WHITE BG
        if reflection:
            # Detect if background is light (white/light gray)
            bg_sample = background.crop((bg_w//2-50, bg_h//2-50, bg_w//2+50, bg_h//2+50))
            avg_brightness = np.array(bg_sample).mean()
            bg_is_light = avg_brightness > 200  # Threshold for light backgrounds
            
            reflection_img = self.create_floor_reflection(
                subject, reflection_intensity, 0.7, reflection_fade, bg_is_light
            )
            
            # Position reflection below the subject with small overlap
            reflection_y = y + sub_h - 10  # Small overlap for natural connection
            
            # Ensure reflection fits in canvas
            if reflection_y < bg_h and reflection_img.height > 0:
                available_height = bg_h - reflection_y
                if reflection_img.height > available_height:
                    # Crop reflection to fit
                    reflection_img = reflection_img.crop((0, 0, reflection_img.width, available_height))
                
                # Paste reflection with proper alpha blending
                if reflection_img.height > 0:
                    canvas.paste(reflection_img, (x, reflection_y), reflection_img)
        
        # Add drop shadow if enabled
        if shadow:
            shadow_img = self.create_drop_shadow(subject, shadow_intensity, shadow_blur, shadow_offset)
            shadow_x, shadow_y = x + shadow_offset[0], y + shadow_offset[1]
            
            # Make sure shadow doesn't overlap with reflection position
            if reflection and shadow_y + subject.height > y + sub_h - 5:
                # Adjust shadow position to avoid overlap with reflection
                shadow_y = min(shadow_y, y + sub_h - 15)
            
            canvas.paste(shadow_img, (shadow_x, shadow_y), shadow_img)
        
        # Paste subject (on top of reflection and shadow)
        canvas.paste(subject, (x, y), subject)
        
        return canvas.convert("RGB")

        
# ------------------------------------------------------------------
# 3️⃣  ENHANCED STREAMLIT APP
# ------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="🎭 AI Studio Pro", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        border-radius: 10px;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🎭 AI Studio Pro")
    st.caption("Professional background removal and studio photography suite")
    
    tab1, tab2 = st.tabs(["🎯 Background Removal", "🎬 Studio Generator"])

    with tab1:
        remove_bg_ui()
    with tab2:
        enhanced_studio_ui()

@st.cache_resource
def get_predictor(path, mtype):
    return EnhancedU2NetPredictor(path, mtype)

def remove_bg_ui():
    st.header("🎯 AI Background Removal")
    
    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Model selection
        st.subheader("🤖 AI Model")
        models = {
            "U²-Net Full (Recommended)": ("ckpt/u2net.pth", "U2NET"),
            "U²-Net Small (Faster)": ("ckpt/u2netp.pth", "U2NETP")
        }
        model_name = st.selectbox("Select Model", list(models.keys()))
        path, mtype = models[model_name]
        
        st.divider()
        
        # Processing options
        st.subheader("🔧 Processing Options")
        enhance = st.checkbox("✨ Enhance Objects", True, help="Improve contrast and sharpness")
        hres = st.checkbox("🔍 High-Resolution Mode", help="Better quality, slower processing")
        
        st.divider()
        
        # Fine-tuning
        st.subheader("🎛️ Fine Tuning")
        thresh = st.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.05, 
                          help="Higher = more conservative detection")
        smooth = st.checkbox("🌊 Smooth Edges", True)
        denoise = st.checkbox("🧹 Remove Noise", True)
        feather = st.checkbox("🪶 Feather Edges", True)
        edge_smooth = st.slider("Edge Smoothness", 0, 5, 2)
        
        st.divider()
        
        # Output options
        st.subheader("📤 Output Options")
        ar = st.selectbox("Aspect Ratio", ["Original", "1:1 Square", "3:4 Portrait", "4:3 Landscape"])
        mode = st.selectbox("Preview Mode", ["Background Removed", "Mask Only", "Side by Side", "All Views"])

    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg", "webp"],
            help="Supported formats: PNG, JPG, JPEG, WebP"
        )
        
        if uploaded_file:
            # Load and display original
            img = Image.open(uploaded_file).convert("RGB")
            
            # Apply aspect ratio cropping
            if ar != "Original":
                crop_ratio = ar.split()[0]  # Get "1:1", "3:4", etc.
                img = ImageCropper.crop(img, crop_ratio)
            
            st.image(img, caption="Original Image", use_container_width=True)
            
            # Show image info
            st.info(f"📐 Size: {img.size[0]} × {img.size[1]} pixels")

    with col2:
        st.subheader("🎯 AI Processing")
        
        if uploaded_file:
            # Check if model exists
            if not os.path.exists(path):
                st.error(f"❌ Model file not found: {path}")
                st.info("Please ensure the model files are in the 'ckpt' folder")
                return
            
            # Process button
            if st.button("🚀 Remove Background", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is working its magic..."):
                    try:
                        # Load model and process
                        predictor = get_predictor(path, mtype)
                        pred = predictor.predict_enhanced(img, enhance, hres)
                        mask = predictor.post_process_mask(pred, thresh, smooth, denoise)
                        result = predictor.remove_bg(img, mask, feather, edge_smooth)
                        
                        # Store result for studio use
                        st.session_state["studio_subject"] = result
                        st.session_state["original_image"] = img
                        
                        # Display results based on mode
                        if mode == "Mask Only":
                            st.image(mask, caption="Generated Mask", use_container_width=True)
                        elif mode == "Background Removed":
                            st.image(result, caption="Background Removed", use_container_width=True)
                        elif mode == "Side by Side":
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.image(img, caption="Before", use_container_width=True)
                            with col_b:
                                st.image(result, caption="After", use_container_width=True)
                        else:  # All Views
                            st.image(mask, caption="Generated Mask", use_container_width=True)
                            st.image(result, caption="Background Removed", use_container_width=True)
                        
                        # Success message
                        st.markdown("""
                        <div class="success-box">
                            ✅ <strong>Success!</strong> Background removed successfully. 
                            Your image is ready for the Studio Generator!
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Download button
                        buf = io.BytesIO()
                        result.save(buf, format="PNG")
                        st.download_button(
                            "📥 Download PNG",
                            data=buf.getvalue(),
                            file_name=f"no_bg_{uploaded_file.name.split('.')[0]}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")
        else:
            st.info("👆 Upload an image to get started")

def enhanced_studio_ui():
    st.header("🎬 Professional Studio Generator")
    
    # Initialize studio generator
    studio = StudioBackgroundGenerator()
    
    # Check if we have a subject from background removal
    has_subject = "studio_subject" in st.session_state
    
    if has_subject:
        st.markdown("""
        <div class="success-box">
            ✅ <strong>Subject Ready!</strong> You have a subject loaded from background removal.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            💡 <strong>Tip:</strong> First remove the background of your image in the "Background Removal" tab, 
            or upload a PNG with transparent background below.
        </div>
        """, unsafe_allow_html=True)
    
    # Three-column layout for better organization
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # ================== SUBJECT MANAGEMENT ==================
    with col1:
        st.subheader("👤 Subject")
        
        # Option to upload new subject
        uploaded_subject = st.file_uploader(
            "Upload PNG with transparent background",
            type=["png"],
            help="Upload a PNG file with transparent background"
        )
        
        if uploaded_subject:
            subject = Image.open(uploaded_subject).convert("RGBA")
            st.session_state["studio_subject"] = subject
            has_subject = True
        
        # Display current subject
        if has_subject:
            subject = st.session_state["studio_subject"]
            st.image(subject, caption="Current Subject", use_container_width=True)
            
            # Subject controls
            st.subheader("📐 Subject Settings")
            subject_scale = st.slider("Size", 0.2, 3.0, 1.0, 0.1, key="subject_scale")
            subject_position = st.selectbox("Position", [
                "Center",
                "Bottom Center", 
                "Top Center",
                "Left Center",
                "Right Center"
            ], key="subject_pos")
        else:
            st.info("No subject loaded. Remove background first or upload a PNG.")
    
    # ================== BACKGROUND CREATION ==================
    with col2:
        st.subheader("🎨 Background")
        
        # Background creation tabs
        bg_tab1, bg_tab2 = st.tabs(["🎨 Presets", "🛠️ Custom"])
        
        with bg_tab1:
            # Preset selection with preview
            preset_name = st.selectbox("Choose Preset", list(studio.gradient_presets.keys()))
            
            # Direction and style
            direction = st.selectbox("Gradient Direction", [
                "Vertical ⬇️",
                "Horizontal ➡️", 
                "Radial 🔄",
                "Diagonal ↘️",
                "Diamond 💎"
            ])
            direction_map = {
                "Vertical ⬇️": "vertical",
                "Horizontal ➡️": "horizontal", 
                "Radial 🔄": "radial",
                "Diagonal ↘️": "diagonal",
                "Diamond 💎": "diamond"
            }
            
            curve = st.selectbox("Gradient Curve", [
                "Linear",
                "Ease In",
                "Ease Out", 
                "Ease In-Out",
                "Bounce"
            ])
            curve_map = {
                "Linear": "linear",
                "Ease In": "ease_in",
                "Ease Out": "ease_out",
                "Ease In-Out": "ease_in_out", 
                "Bounce": "bounce"
            }
        
        with bg_tab2:
            # Custom colors
            st.write("Custom Gradient Colors")
            custom_color1 = st.color_picker("Color 1", "#FFFFFF")
            custom_color2 = st.color_picker("Color 2", "#F0F0F0")
            
            # Convert hex to RGB
            def hex_to_rgb(hex_color):
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
            if st.button("Use Custom Colors"):
                studio.gradient_presets["🎨 Custom"] = [hex_to_rgb(custom_color1), hex_to_rgb(custom_color2)]
                preset_name = "🎨 Custom"
        
        # Background size
        size_preset = st.selectbox("Output Size", [
            "1080×1080 (Instagram Square)",
            "1080×1350 (Instagram Portrait)", 
            "1920×1080 (HD Landscape)",
            "2048×2048 (High Resolution)",
            "Custom Size"
        ])
        
        if size_preset == "Custom Size":
            col_w, col_h = st.columns(2)
            with col_w:
                custom_width = st.number_input("Width", 100, 4000, 1080)
            with col_h:
                custom_height = st.number_input("Height", 100, 4000, 1080)
            size = (custom_width, custom_height)
        else:
            size_map = {
                "1080×1080 (Instagram Square)": (1080, 1080),
                "1080×1350 (Instagram Portrait)": (1080, 1350),
                "1920×1080 (HD Landscape)": (1920, 1080),
                "2048×2048 (High Resolution)": (2048, 2048)
            }
            size = size_map[size_preset]
        
        # Generate background button
        if st.button("🎨 Generate Background", type="primary", use_container_width=True):
            with st.spinner("Creating beautiful background..."):
                colors = studio.gradient_presets[preset_name]
                bg = studio.create_gradient_background(
                    size, colors, 
                    direction_map[direction], 
                    curve_map[curve]
                )
                st.session_state["studio_background"] = bg
        
        # Display current background
        if "studio_background" in st.session_state:
            bg = st.session_state["studio_background"]
            st.image(bg, caption="Generated Background", use_container_width=True)
    
    # ================== EFFECTS & COMPOSITION ==================
    with col3:
        st.subheader("✨ Effects & Final")
        
        # Effects section
        st.write("**Lighting & Shadow Effects**")
        
        # Shadow settings
        enable_shadow = st.checkbox("🌑 Drop Shadow", True)
        if enable_shadow:
            shadow_intensity = st.slider("Shadow Opacity", 0.0, 1.0, 0.3, 0.1)
            shadow_blur = st.slider("Shadow Blur", 1, 20, 5)
        else:
            shadow_intensity = 0
            shadow_blur = 5
        
        # TAMBAHAN BARU: Floor Reflection Settings
        st.divider()
        st.write("**🪞 Floor Reflection**")
        enable_reflection = st.checkbox("🪞 Floor Reflection", False, 
                                    help="Add realistic floor reflection effect")
        if enable_reflection:
            reflection_intensity = st.slider("Reflection Strength", 0.0, 1.0, 0.4, 0.05,
                                        help="How strong the reflection appears")
            reflection_fade = st.slider("Reflection Height", 0.2, 1.0, 0.6, 0.1,
                                    help="How much of the reflection to show")
            
            # Reflection style
            reflection_style = st.selectbox("Reflection Style", [
                "Natural",
                "Subtle", 
                "Dramatic"
            ])
            
            # Auto-adjust based on style
            if reflection_style == "Subtle":
                reflection_intensity = min(reflection_intensity, 0.3)
                reflection_fade = min(reflection_fade, 0.4)
            elif reflection_style == "Dramatic":
                reflection_intensity = max(reflection_intensity, 0.5)
                reflection_fade = max(reflection_fade, 0.7)
        else:
            reflection_intensity = 0
            reflection_fade = 0.6
            reflection_style = "Natural"
        
        # Lighting effects
        lighting_effect = st.selectbox("💡 Lighting", [
            "None",
            "Top Light",
            "Center Spotlight"
        ])
        lighting_map = {"None": "none", "Top Light": "top", "Center Spotlight": "center"}
        
        if lighting_effect != "None":
            lighting_intensity = st.slider("Lighting Intensity", 0.0, 0.8, 0.2, 0.1)
        else:
            lighting_intensity = 0
        
        st.divider()
        
        # Final composition
        st.write("**🎬 Final Composition**")
        
        # Check if we can compose
        can_compose = has_subject and "studio_background" in st.session_state
        
        if can_compose:
            if st.button("🎬 Create Studio Photo", type="primary", use_container_width=True):
                with st.spinner("🎭 Creating your studio masterpiece..."):
                    subject = st.session_state["studio_subject"]
                    background = st.session_state["studio_background"]
                    
                    # Create composition with reflection
                    final_image = studio.composite_with_background(
                        subject=subject,
                        background=background,
                        position=subject_position.lower().replace(" ", "_"),
                        scale=subject_scale,
                        shadow=enable_shadow,
                        shadow_intensity=shadow_intensity,
                        shadow_blur=shadow_blur,
                        lighting=lighting_map[lighting_effect],
                        lighting_intensity=lighting_intensity,
                        reflection=enable_reflection,
                        reflection_intensity=reflection_intensity,
                        reflection_fade=reflection_fade
                    )
                    
                    st.session_state["final_composition"] = final_image
            
            # Display final result
            if "final_composition" in st.session_state:
                final = st.session_state["final_composition"]
                st.image(final, caption="✨ Studio Photo", use_container_width=True)
                
                # Download options
                st.write("**📥 Download Options**")
                
                # High quality PNG
                buf_png = io.BytesIO()
                final.save(buf_png, format="PNG", optimize=True)
                st.download_button(
                    "📥 PNG (Best Quality)",
                    data=buf_png.getvalue(),
                    file_name="studio_photo.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                # Compressed JPEG
                buf_jpg = io.BytesIO()
                final.save(buf_jpg, format="JPEG", quality=95, optimize=True)
                st.download_button(
                    "📥 JPEG (Smaller Size)",
                    data=buf_jpg.getvalue(),
                    file_name="studio_photo.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
                
        else:
            missing_items = []
            if not has_subject:
                missing_items.append("Subject")
            if "studio_background" not in st.session_state:
                missing_items.append("Background")
            
            st.info(f"📋 Need: {', '.join(missing_items)}")
    
    # ================== BATCH PROCESSING SECTION ==================
    st.divider()
    st.subheader("🔄 Batch Processing")
    
    batch_col1, batch_col2 = st.columns([2, 1])
    
    with batch_col1:
        st.write("**📁 Upload Multiple Images**")
        batch_files = st.file_uploader(
            "Upload multiple PNG files with transparent backgrounds",
            type=["png"],
            accept_multiple_files=True,
            help="Select multiple PNG files to process with the same background"
        )
        
        if batch_files:
            st.success(f"✅ {len(batch_files)} files uploaded")
            
            # Preview first few images
            if len(batch_files) > 0:
                st.write("**👀 Preview**")
                preview_cols = st.columns(min(4, len(batch_files)))
                for i, file in enumerate(batch_files[:4]):
                    with preview_cols[i]:
                        img = Image.open(file).convert("RGBA")
                        st.image(img, caption=file.name[:15] + "...", use_container_width=True)
                
                if len(batch_files) > 4:
                    st.info(f"+ {len(batch_files) - 4} more files...")
    
    with batch_col2:
        st.write("**⚙️ Batch Settings**")
        
        if "studio_background" in st.session_state:
            st.success("✅ Background ready")
        else:
            st.warning("⚠️ Generate background first")
        
        # Batch settings
        batch_preset = st.selectbox("Background Preset", list(studio.gradient_presets.keys()))
        batch_position = st.selectbox("Position", ["center", "bottom_center", "top_center"])
        batch_scale = st.slider("Scale", 0.2, 2.0, 1.0, 0.1, key="batch_scale")
        batch_shadow = st.checkbox("Shadow", True, key="batch_shadow")
        
        # TAMBAHAN BARU: Batch reflection settings
        batch_reflection = st.checkbox("Floor Reflection", False, key="batch_reflection")
        if batch_reflection:
            batch_reflection_intensity = st.slider("Reflection Strength", 0.1, 0.8, 0.4, 0.1, key="batch_refl_int")
        else:
            batch_reflection_intensity = 0.4
        
        # Process batch - UPDATE bagian ini untuk include reflection
        if batch_files and st.button("🚀 Process Batch", type="primary", use_container_width=True):
            # Use existing background or create new one
            if "studio_background" in st.session_state:
                batch_bg = st.session_state["studio_background"]
            else:
                batch_bg = studio.create_gradient_background(
                    (1080, 1080), 
                    studio.gradient_presets[batch_preset],
                    "vertical", 
                    "linear"
                )
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create ZIP file
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for i, file in enumerate(batch_files):
                    status_text.text(f"Processing {file.name}...")
                    progress_bar.progress((i + 1) / len(batch_files))
                    
                    # Load and process image
                    subject = Image.open(file).convert("RGBA")
                    
                    # Create composition with reflection
                    result = studio.composite_with_background(
                        subject=subject,
                        background=batch_bg,
                        position=batch_position,
                        scale=batch_scale,
                        shadow=batch_shadow,
                        shadow_intensity=0.3,
                        reflection=batch_reflection,
                        reflection_intensity=batch_reflection_intensity,
                        reflection_fade=0.6
                    )
                    
                    # Save to ZIP
                    img_buffer = io.BytesIO()
                    result.save(img_buffer, format="PNG")
                    zip_file.writestr(f"studio_{file.name}", img_buffer.getvalue())
            
            status_text.text("✅ Batch processing complete!")
            
            # Download ZIP
            st.download_button(
                "📦 Download Batch Results (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="studio_batch_results.zip",
                mime="application/zip",
                use_container_width=True
            )
    
    # ================== QUICK TEMPLATES SECTION ==================
    st.divider()
    st.subheader("⚡ Quick Templates")
    st.write("One-click professional setups")
    
    template_cols = st.columns(4)
    
    templates = {
        "📸 Portrait Studio": {
            "preset": "🤍 Professional White",
            "direction": "vertical",
            "position": "center",
            "scale": 1.0,
            "shadow": True,
            "lighting": "top",
            "reflection": False,
            "reflection_intensity": 0.3
        },
        "👗 Fashion Shoot": {
            "preset": "💗 Fashion Pink", 
            "direction": "vertical",
            "position": "center",
            "scale": 1.1,
            "shadow": True,
            "lighting": "center",
            "reflection": True,
            "reflection_intensity": 0.4
        },
        "💼 Corporate": {
            "preset": "🩶 Soft Gray",
            "direction": "vertical", 
            "position": "center",
            "scale": 0.9,
            "shadow": False,
            "lighting": "none",
            "reflection": False,
            "reflection_intensity": 0.2
        },
        "🌟 Luxury": {
            "preset": "🖤 Luxury Black",
            "direction": "radial",
            "position": "center", 
            "scale": 1.2,
            "shadow": True,
            "lighting": "center",
            "reflection": True,
            "reflection_intensity": 0.5
        }
    }
    
    for i, (template_name, settings) in enumerate(templates.items()):
        with template_cols[i]:
            if st.button(template_name, use_container_width=True):
                if has_subject:
                    with st.spinner(f"Applying {template_name} template..."):
                        # Create background with template settings
                        bg = studio.create_gradient_background(
                            (1080, 1080),
                            studio.gradient_presets[settings["preset"]],
                            settings["direction"],
                            "ease_in_out"
                        )
                        
                        # Create composition with reflection
                        subject = st.session_state["studio_subject"]
                        result = studio.composite_with_background(
                            subject=subject,
                            background=bg,
                            position=settings["position"],
                            scale=settings["scale"],
                            shadow=settings["shadow"],
                            shadow_intensity=0.3,
                            lighting=settings["lighting"],
                            lighting_intensity=0.2,
                            reflection=settings["reflection"],
                            reflection_intensity=settings["reflection_intensity"],
                            reflection_fade=0.6
                        )
                        
                        st.session_state["final_composition"] = result
                        st.success(f"✅ {template_name} applied!")
                        st.rerun()
                else:
                    st.warning("Upload or create a subject first!")
    
    # ================== TIPS & HELP SECTION ==================
    with st.expander("💡 Tips & Help"):
        st.markdown("""
        ### 🎯 **Getting Started**
        1. **Remove Background**: Use the Background Removal tab to extract your subject
        2. **Create Background**: Choose a preset or create custom gradients
        3. **Compose**: Adjust position, scale, and effects
        4. **Add Effects**: Try shadows, lighting, and floor reflections
        5. **Download**: Get your professional studio photo!

        ### 🎨 **Background Tips**
        - **White/Gray**: Professional portraits, corporate headshots
        - **Warm tones**: Fashion, lifestyle photography  
        - **Cool tones**: Tech products, modern aesthetics
        - **Dark backgrounds**: Luxury items, dramatic portraits

        ### ✨ **Effect Guidelines**
        - **Drop Shadow**: Adds depth and realism (subtle works best)
        - **Top Lighting**: Simulates studio softbox lighting
        - **Center Spotlight**: Creates dramatic focus on subject
        - **Floor Reflection**: Adds luxury feel, best for products and fashion

        ### 🪞 **Reflection Tips**
        - **Natural (0.3-0.4)**: Subtle, professional look
        - **Dramatic (0.5-0.7)**: High-end product photography
        - **Height 0.4-0.6**: Most realistic appearance
        - **Works best**: Products, fashion items, portraits on dark/glossy backgrounds

        ### 📐 **Composition Tips**
        - **Center**: Balanced, professional look
        - **Bottom Center**: Shows reflection naturally, more dynamic
        - **Scale 0.8-1.2**: Most natural looking range
        - **Reflection + Shadow**: Ultimate realism (use moderate settings)
        """)

if __name__ == "__main__":
    main()