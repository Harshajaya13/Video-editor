from moviepy.editor import VideoFileClip, concatenate_videoclips
import step3_brain as brain

# CONFIGURATION
INPUT_VIDEO = "test_video.mp4"
OUTPUT_VIDEO = "final_summary.mp4"

def render_video():
    print("-" * 30)
    print("🎬 CUTTER: Starting Video Render...")
    
    # 1. Get the timestamps from The Brain
    # This runs step 1, 2, and 3 automatically
    segments = brain.get_highlights(INPUT_VIDEO)
    
    if not segments:
        print("❌ Error: Brain returned no segments.")
        return

    # 2. Load the original video
    clip = VideoFileClip(INPUT_VIDEO)
    
    # 3. Cut the clips
    final_clips = []
    print(f"\n✂️ Processing {len(segments)} cuts...")
    
    for start, end in segments:
        # Buffer: Add 0.1s padding so audio doesn't sound cut off
        # Clamp to ensure we don't go past video limits
        start_clean = max(0, start)
        end_clean = min(clip.duration, end)
        
        print(f"   Cutting: {start_clean:.2f} to {end_clean:.2f}")
        
        sub = clip.subclip(start_clean, end_clean)
        final_clips.append(sub)
        
    # 4. Stitch them together
    print("\n🧵 Stitching clips together...")
    final_cut = concatenate_videoclips(final_clips)
    
    # 5. Save the file
    print(f"💾 Saving to {OUTPUT_VIDEO} (This takes time)...")
    # preset='ultrafast' makes it render quickly for testing
    final_cut.write_videofile(OUTPUT_VIDEO, codec="libx264", audio_codec="aac", preset="ultrafast")
    
    print(f"\n✅ DONE! Open {OUTPUT_VIDEO} to see your AI masterpiece.")

if __name__ == "__main__":
    render_video()
