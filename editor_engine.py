import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips
import numpy as np

def process_and_render(video_path, df, base_fade=0.1):
    """
    Cuts video based on 'ai_decision'.
    Applies ADAPTIVE FADES (Effects) based on volume intensity.
    """
    print(f"🎬 ENGINE START: Processing {video_path}...")
    
    try:
        original_clip = VideoFileClip(video_path)
    except Exception as e:
        print(f"❌ Error loading video: {e}")
        return None

    # 1. Identify "Keep" Segments
    keep_indices = df[df['ai_decision'] == 1].index.tolist()
    if not keep_indices:
        print("❌ No segments to keep.")
        return None

    # Group continuous frames into ranges
    ranges = []
    start = keep_indices[0]
    for i in range(1, len(keep_indices)):
        if keep_indices[i] != keep_indices[i-1] + 1:
            ranges.append((start, keep_indices[i-1]))
            start = keep_indices[i]
    ranges.append((start, keep_indices[-1]))

    print(f"✂️ Found {len(ranges)} cuts to process...")

    # 2. Process Clips with SMART EFFECTS
    sub_clips = []
    fps = original_clip.fps
    
    for i, (start_idx, end_idx) in enumerate(ranges):
        start_t = df.loc[start_idx, 'timestamp']
        # Add a tiny buffer (one frame) to ensure smoothness
        end_t = df.loc[end_idx, 'timestamp'] + (1.0/fps)
        
        clip = original_clip.subclip(start_t, end_t)
        
        # --- 🧠 SMART FADE LOGIC ---
        # "If loud -> Slow Fade. If quiet -> Fast Cut."
        
        # Get average volume of this specific clip
        clip_volume = df.loc[start_idx:end_idx, 'rms_volume'].mean()
        
        current_fade = base_fade
        
        # If the clip is loud (>0.1 RMS), use a smoother fade (0.3s)
        # This creates that "Professional Transition" feel
        if clip_volume > 0.1: 
            current_fade = 0.3
        
        # Apply Audio Fade (The Effect)
        clip = clip.audio_fadein(current_fade).audio_fadeout(current_fade)
        
        sub_clips.append(clip)

    # 3. Stitch and Render
    if sub_clips:
        print("🔗 Stitching clips...")
        output_filename = "final_smart_render.mp4"
        
        final_video = concatenate_videoclips(sub_clips, method="compose")
        final_video.write_videofile(
            output_filename, 
            codec="libx264", 
            audio_codec="aac", 
            fps=fps,
            preset="medium" # Good balance of speed and quality
        )
        return output_filename
    else:
        return None
