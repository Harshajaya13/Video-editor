from moviepy.editor import VideoFileClip, concatenate_videoclips
import pandas as pd
import os

def process_and_render(video_path, df, output_path="final_edit.mp4"):
    """
    Takes the video and the Decision Dataframe.
    Cuts the video based on 'ai_decision' column (1 = Keep, 0 = Cut).
    Safely handles missing volume data.
    """
    
    # 1. VALIDATION
    if 'ai_decision' not in df.columns:
        print("❌ Error: Dataframe is missing 'ai_decision' column.")
        return None

    # Check if we have any '1's (Keep)
    keep_indices = df[df['ai_decision'] == 1].index.tolist()
    if not keep_indices:
        print("❌ AI decided to cut the whole video.")
        return None

    print(f"🎬 Loading Video: {video_path}")
    try:
        original_video = VideoFileClip(video_path)
    except Exception as e:
        print(f"❌ Error loading video file: {e}")
        return None

    # 2. GROUPING INDICES INTO TIME SEGMENTS
    segments = []
    if keep_indices:
        start_idx = keep_indices[0]
        prev_idx = keep_indices[0]

        for idx in keep_indices[1:]:
            # Check if indices are consecutive
            if idx == prev_idx + 1:
                prev_idx = idx
            else:
                segments.append((start_idx, prev_idx))
                start_idx = idx
                prev_idx = idx
        segments.append((start_idx, prev_idx))

    # 3. SMART EDITING LOOP
    clips_to_join = []
    previous_vol = 0.5 
    
    # Check if volume data exists to allow smart fading
    has_volume_data = 'rms_volume' in df.columns
    
    print(f"✂️  Cutting {len(segments)} segments...")

    for i, (start_row, end_row) in enumerate(segments):
        # Get Timestamps
        start_time = df.loc[start_row, 'timestamp']
        # Add 0.5s buffer but don't go past video end
        end_time = min(df.loc[end_row, 'timestamp'] + 0.5, original_video.duration)
        
        # Extract the Clip
        clip = original_video.subclip(start_time, end_time)

        # --- SAFE FADE LOGIC ---
        if has_volume_data:
            # Get average volume of THIS specific clip
            current_vol = df.loc[start_row:end_row, 'rms_volume'].mean()

            # Logic: Fade in if we jump from Silence -> Loud
            is_quiet_before = previous_vol < 0.05
            is_loud_now = current_vol > 0.1

            if i > 0 and (is_quiet_before and is_loud_now):
                print(f"   ✨ Smart Fade-In at {start_time}s")
                clip = clip.fadein(1.0).audio_fadein(1.0)
            else:
                # Standard smooth cut
                clip = clip.audio_fadein(0.1).audio_fadeout(0.1)
            
            previous_vol = current_vol
        else:
            # Fallback: Just do smooth cuts if no volume data
            clip = clip.audio_fadein(0.1).audio_fadeout(0.1)

        clips_to_join.append(clip)

    # 4. RENDER FINAL VIDEO
    if clips_to_join:
        print("💾 Rendering final file...")
        try:
            final_video = concatenate_videoclips(clips_to_join, method="compose")
            
            final_video.write_videofile(
                output_path, 
                codec="libx264", 
                audio_codec="aac", 
                fps=original_video.fps,  # <--- CRITICAL FIX INCLUDED
                logger="bar" # Shows a progress bar
            )
            print("✅ Done!")
            
            # Close the clip to release memory
            original_video.close()
            return output_path
            
        except Exception as e:
            print(f"❌ Error during rendering: {e}")
            return None
