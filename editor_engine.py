from moviepy.editor import VideoFileClip, concatenate_videoclips
import pandas as pd

def process_and_render(video_path, df, output_path="final_edit.mp4"):
    """
    1. Groups consecutive '1's into clips.
    2. Checks volume changes between clips.
    3. Applies Fades if the mood changes too fast.
    """
    original_video = VideoFileClip(video_path)
    clips_to_join = []
    
    # --- 1. GROUPING LOGIC (Turn rows into Time Segments) ---
    # We turn [0, 1, 1, 1, 0, 1, 1] into [(0.5, 2.0), (3.0, 4.0)]
    keep_indices = df[df['ai_decision'] == 1].index.tolist()
    
    if not keep_indices:
        return None # AI rejected everything

    segments = []
    start_idx = keep_indices[0]
    prev_idx = keep_indices[0]

    for idx in keep_indices[1:]:
        if idx == prev_idx + 1:
            prev_idx = idx # Continue the segment
        else:
            segments.append((start_idx, prev_idx)) # Close segment
            start_idx = idx
            prev_idx = idx
    segments.append((start_idx, prev_idx)) # Close final segment

    # --- 2. EDITING LOGIC (The "Smart Fades") ---
    previous_rms = 0.5 # Default start value
    
    print(f"🎬 Editing {len(segments)} segments...")

    for start_row, end_row in segments:
        # Get Time & Volume Data
        start_time = df.loc[start_row, 'timestamp']
        # Add 0.5s buffer to end time so it doesn't cut abruptly
        end_time = df.loc[end_row, 'timestamp'] + 0.5 
        
        # Calculate Average Volume of THIS specific segment
        segment_rms = df.loc[start_row:end_row, 'rms_volume'].mean()

        # Create the basic cut
        clip = original_video.subclip(start_time, end_time)

        # --- THE FADE CHECK ---
        # Logic: If we go from Quiet (Low RMS) -> Loud (High RMS), Fade In.
        # Otherwise, just do a hard Jump Cut (it feels faster/viral).
        
        is_quiet_before = previous_rms < 0.02
        is_loud_now = segment_rms > 0.1
        
        if is_quiet_before and is_loud_now:
            print(f"   ✨ Adding Fade In at {start_time}s (Quiet -> Loud)")
            clip = clip.fadein(1.0) # 1 Second smooth entry
        else:
            # Optional: Add a tiny 0.1s crossfade just to prevent audio "pops"
            clip = clip.audio_fadein(0.05).audio_fadeout(0.05)

        clips_to_join.append(clip)
        
        # Update tracker
        previous_rms = segment_rms

    # --- 3. RENDERING ---
    print("💾 Combining clips...")
    final_video = concatenate_videoclips(clips_to_join, method="compose")
    
    # We write to a specific file
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
    
    return output_path
