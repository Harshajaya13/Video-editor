import pandas as pd
import tempfile
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx

def process_and_render(video_path, df, base_fade=0.1):
    print(f"🎬 ENGINE START: Processing {video_path}...")
    
    try:
        # Load Video
        original_clip = VideoFileClip(video_path)
        fps = original_clip.fps
        duration = original_clip.duration
        
        # Safety Check: If FPS is broken, default to 30
        if fps is None or fps == 0:
            fps = 30
            
    except Exception as e:
        print(f"❌ Error loading video: {e}")
        return None

    # 1. Identify "Keep" Segments
    # We sort by timestamp to ensure cuts are in order
    df = df.sort_values('timestamp')
    keep_indices = df[df['ai_decision'] == 1].index.tolist()
    
    if not keep_indices:
        print("❌ No segments to keep.")
        return None

    # Group continuous frames into ranges (timestamps)
    ranges = []
    if keep_indices:
        start_idx = keep_indices[0]
        prev_idx = keep_indices[0]
        
        for i in range(1, len(keep_indices)):
            curr_idx = keep_indices[i]
            # If the gap between frames is > 1 (meaning a cut), save the range
            if curr_idx != prev_idx + 1:
                ranges.append((start_idx, prev_idx))
                start_idx = curr_idx
            prev_idx = curr_idx
        ranges.append((start_idx, prev_idx))

    print(f"✂️ Found {len(ranges)} cuts to process...")

    # 2. Process Clips with SMART EFFECTS
    sub_clips = []
    
    for i, (start_idx, end_idx) in enumerate(ranges):
        # Get start/end time from dataframe
        # We add buffer to end_t to prevent "glitchy" audio at the very edge
        start_t = df.loc[start_idx, 'timestamp']
        end_t = df.loc[end_idx, 'timestamp'] + (1.0/fps) # Add 1 frame buffer
        
        # Clamp to video duration
        start_t = max(0, start_t)
        end_t = min(duration, end_t)
        
        # Skip invalid clips (too short)
        if end_t - start_t < 0.2:
            continue
            
        clip = original_clip.subclip(start_t, end_t)
        
        # --- 🧠 SMART FADE LOGIC ---
        # If the clip is LOUD, we need a slower fade so it doesn't "pop".
        # If the clip is quiet, we can cut fast.
        
        # Calculate volume for this specific segment
        try:
            # Safe slice
            clip_rows = df.loc[start_idx:end_idx]
            if 'rms_volume' in clip_rows:
                clip_volume = clip_rows['rms_volume'].mean()
            else:
                clip_volume = 0
                
            current_fade = base_fade
            if clip_volume > 0.15: 
                current_fade = 0.3 # Smoother fade for loud clips
                
            # Apply Audio Fades
            clip = clip.audio_fadein(current_fade).audio_fadeout(current_fade)
        except:
            # Fallback if audio analysis fails
            pass
            
        sub_clips.append(clip)

    # 3. Stitch and Render
    if sub_clips:
        print("🔗 Stitching clips (This takes time)...")
        
        # Generate Temp File
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_filename = tfile.name
        tfile.close() # Close so MoviePy can write to it
        
        final_video = concatenate_videoclips(sub_clips, method="compose")
        
        # WRITE THE FILE
        # We explicitly enforce FPS and Codec to fix the "2x Speed" bug
        final_video.write_videofile(
            output_filename, 
            codec="libx264", 
            audio_codec="aac", 
            fps=fps,               # <--- This fixes the Speed Up issue
            preset="ultrafast",    # <--- Makes rendering faster
            threads=4,
            logger=None            # Hides the noisy progress bar in terminal
        )
        
        original_clip.close()
        return output_filename
    else:
        original_clip.close()
        return None
