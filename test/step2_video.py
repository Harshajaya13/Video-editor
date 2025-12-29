import cv2
import numpy as np
import os

# CONFIGURATION
VIDEO_PATH = "test_video.mp4"

def analyze_video_motion(video_path):
    print(f"1. Opening video: {video_path}...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")

    motion_scores = []
    prev_frame = None
    
    print("2. Scanning for motion (this checks every frame)...")

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # OPTIMIZATION: Resize to 320px width. 
        # Processing 4K/1080p is too slow and unnecessary for motion detection.
        height, width = frame.shape[:2]
        new_width = 320
        new_height = int(height * (new_width / width))
        frame_small = cv2.resize(frame, (new_width, new_height))

        # Convert to Grayscale (Color doesn't matter for motion)
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        
        # If this is the first frame, just store it and skip
        if prev_frame is None:
            prev_frame = gray
            continue
            
        # CALCULATE DIFFERENCE (The Core Logic)
        # absdiff = |CurrentFrame - PrevFrame|
        frame_diff = cv2.absdiff(prev_frame, gray)
        
        # Sum of all white pixels (amount of change)
        score = np.sum(frame_diff)
        motion_scores.append(score)
        
        # Update previous frame
        prev_frame = gray
        
        # Show progress every 100 frames
        if frame_count % 100 == 0:
            print(f"   Processed {frame_count}/{total_frames} frames...", end='\r')

    cap.release()
    print("\n   Processing Complete.")

    # NORMALIZE (0 to 1)
    # We convert list to numpy array for fast math
    scores = np.array(motion_scores)
    
    if len(scores) > 0:
        scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        
        print("\n✅ SUCCESS: Video Analysis Complete")
        print("-" * 30)
        print(f"First 10 Motion Values: {scores_norm[:10]}")
        print(f"Max Motion found at frame: {np.argmax(scores_norm)}")
        return scores_norm, fps
    else:
        print("❌ Error: No frames processed.")
        return [], 0

if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print("⚠️ Video file not found.")
    else:
        analyze_video_motion(VIDEO_PATH)
