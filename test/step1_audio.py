import librosa
import numpy as np
from moviepy.editor import VideoFileClip
import os

# CONFIGURATION
VIDEO_PATH = "test_video.mp4"  
AUDIO_TEMP = "temp_audio.wav"

def extract_audio_features(video_path):
    print(f"1. Extracting audio from {video_path}...")
    
    # A. Extract Audio using MoviePy
    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(AUDIO_TEMP, verbose=False, logger=None)
    except OSError:
        print("❌ Error: File not found or FFmpeg issue.")
        return

    print("2. Analyzing Audio (this takes a moment)...")
    
    # B. Load into Librosa
    # sr=None preserves original sampling rate
    y, sr = librosa.load(AUDIO_TEMP, sr=None)
    
    # C. Calculate Features
    # RMS = Volume / Energy
    rms = librosa.feature.rms(y=y)[0]
    
    # ZCR = Zero Crossing Rate (Sharp noises like hits, clicks, fast speech)
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    
    # D. Normalize Features (Scale them 0 to 1)
    # This is CRITICAL so we can compare them later
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    rms_norm = normalize(rms)
    zcr_norm = normalize(zcr)
    
    # Generate time axis
    times = librosa.times_like(rms, sr=sr)
    
    print("\n✅ SUCCESS: Audio Analysis Complete")
    print("-" * 30)
    print(f"Total Duration: {times[-1]:.2f} seconds")
    print(f"Data Points: {len(rms_norm)}")
    print(f"Avg Volume: {np.mean(rms_norm):.4f}")
    print(f"Max Volume: {np.max(rms_norm):.4f}")
    
    # Cleanup temp file
    if os.path.exists(AUDIO_TEMP):
        os.remove(AUDIO_TEMP)
        
    return times, rms_norm, zcr_norm

if __name__ == "__main__":
    # Create a dummy video file if you don't have one, just to test imports
    if not os.path.exists(VIDEO_PATH):
        print(f"⚠️ WARNING: {VIDEO_PATH} not found. Please put a video file in this folder.")
    else:
        times, vol, sharp = extract_audio_features(VIDEO_PATH)
        # Show first 10 values to prove it works
        print("\nFirst 10 Volume Values:", vol[:10])
