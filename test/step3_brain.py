
import numpy as np
import step1_audio as audio_tool
import step2_video as video_tool

# CONFIGURATION
VIDEO_PATH = "test_video.mp4"

def get_highlights(video_path):
    print("🧠 BRAIN: Starting Analysis...")
    
    # 1. GET DATA
    times, audio_scores, _ = audio_tool.extract_audio_features(video_path)
    video_scores, fps = video_tool.analyze_video_motion(video_path)
    
    # 2. ALIGNMENT
    print(f"   Audio points: {len(audio_scores)} | Video points: {len(video_scores)}")
    
    # Interpolate Video to match Audio length
    original_video_x = np.linspace(0, len(audio_scores), len(video_scores))
    target_audio_x = np.arange(len(audio_scores))
    video_scores_aligned = np.interp(target_audio_x, original_video_x, video_scores)
    
    # 3. FUSION (The Recipe)
    final_scores = (audio_scores * 0.7) + (video_scores_aligned * 0.3)
    
    # 4. INTELLIGENT THRESHOLDING
    avg_score = np.mean(final_scores)
    
    # LOWERED THE BAR: Just use Average. If it's above average, it's good.
    threshold = avg_score 
    
    print(f"   Threshold set to: {threshold:.4f}")
    
    # 5. EXTRACT TIMESTAMPS
    keep_mask = final_scores > threshold
    segments = []
    start_time = None
    
    for i, is_keep in enumerate(keep_mask):
        current_time = times[i]
        if is_keep and start_time is None:
            start_time = current_time
        elif not is_keep and start_time is not None:
            end_time = current_time
            # CHANGED: Min duration lowered to 0.5s for short videos
            if (end_time - start_time) > 0.5:
                segments.append((start_time, end_time))
            start_time = None

    # 6. FAILSAFE (The "Winner" Logic)
    # If no clips found, force it to return the middle 30% of the video
    if len(segments) == 0:
        print("⚠️ WARNING: No clear highlights found. Engaging Failsafe Mode.")
        total_duration = times[-1]
        start = total_duration * 0.3
        end = total_duration * 0.7
        segments.append((start, end))

    print(f"\n✅ DECISION MADE: Found {len(segments)} Highlight Clips.")
    for i, (s, e) in enumerate(segments[:5]):
        print(f"   Clip {i+1}: {s:.2f}s -> {e:.2f}s")
        
    return segments

if __name__ == "__main__":
    segments = get_highlights(VIDEO_PATH)
