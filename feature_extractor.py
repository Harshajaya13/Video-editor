import cv2
import numpy as np
import pandas as pd
import librosa
from deepface import DeepFace
import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

def extract_all_features(video_path, output_csv="raw_features_for_rohit.csv", step_seconds=1.0):
    """
    SUPER HYBRID EXTRACTOR:
    Combines 'Old Engineering Metrics' (Pitch, Motion) 
    with 'New AI Metrics' (Emotions).
    """
    if not os.path.exists(video_path):
        print("❌ Video file not found.")
        return None

    print(f"🎬 Starting SUPER HYBRID analysis on: {video_path}")
    
    # --- 1. AUDIO SETUP ---
    try:
        y, sr = librosa.load(video_path, sr=None)
    except Exception:
        y, sr = None, None

    # --- 2. VIDEO SETUP ---
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_step = int(fps * step_seconds)
    
    data = []
    current_frame = 0
    prev_gray = None # To calculate motion

    print("🧠 Extracting: [Pitch/Motion] (Old) + [Emotions] (New)...")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret: break

        timestamp = current_frame / fps
        
        # --- A. VISUAL METRICS ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate Motion (Difference between this frame and previous frame)
        motion_score = 0.0
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_score = np.mean(diff) / 255.0
        prev_gray = gray
        
        # --- B. AUDIO METRICS ---
        rms_volume = 0
        zcr_pitch = 0 # Zero Crossing Rate (Proxy for pitch/energy)
        
        if y is not None:
            start = int(timestamp * sr)
            end = int((timestamp + step_seconds) * sr)
            if end < len(y):
                chunk = y[start:end]
                # Volume
                rms_volume = float(np.sqrt(np.mean(chunk**2)))
                # Pitch/Frequency (ZCR)
                zcr_pitch = float(np.mean(librosa.feature.zero_crossing_rate(chunk)))

        # --- C. EMOTION METRICS ---
        # Defaults
        dom_emotion = "neutral"
        p_happy, p_surprise, p_sad, p_neutral = 0.0, 0.0, 0.0, 0.0
        
        try:
            # Fast DeepFace Analysis
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list): analysis = analysis[0]
            
            scores = analysis['emotion']
            dom_emotion = analysis['dominant_emotion']
            
            p_happy = scores.get('happy', 0) / 100.0
            p_surprise = scores.get('surprise', 0) / 100.0
            p_sad = scores.get('sad', 0) / 100.0
            p_neutral = scores.get('neutral', 0) / 100.0
            
            print(f"⏱️ {timestamp:.1f}s | Mot: {motion_score:.3f} | Vol: {rms_volume:.3f} | Emo: {dom_emotion}")
            
        except Exception:
            pass 

        # --- D. BUILD THE HYBRID ROW ---
        row = {
            # 1. The OLD Columns (Essential for structure)
            "timestamp": round(timestamp, 2),
            "rms_volume": round(rms_volume, 4),
            "zcr_pitch": round(zcr_pitch, 4),      # <--- Restored!
            "motion_score": round(motion_score, 4), # <--- Restored!
            "brightness": round(brightness, 3),
            
            # 2. The NEW Columns (Added for better AI)
            "blur_check": round(blur_score, 1),
            "dominant_emotion": dom_emotion,
            "prob_happy": p_happy,
            "prob_surprise": p_surprise,
            "prob_sad": p_sad,
            "prob_neutral": p_neutral
        }
        data.append(row)
        current_frame += frame_step

    cap.release()
    
    # Organize columns strictly
    cols = [
        "timestamp", "rms_volume", "zcr_pitch", "motion_score", "brightness", # OLD GROUP
        "blur_check", "dominant_emotion", "prob_happy", "prob_surprise", "prob_sad", "prob_neutral" # NEW GROUP
    ]
    
    df = pd.DataFrame(data)
    # Ensure only valid columns are written, in correct order
    df = df[cols]
    
    df.to_csv(output_csv, index=False)
    print(f"✅ HYBRID DATA SAVED: {output_csv}")
    print(f"   Includes: Motion, Pitch, Volume, Brightness + EMOTIONS")
    return df

if __name__ == "__main__":
    extract_all_features("input_video.mp4")
