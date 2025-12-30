import cv2
import numpy as np
import pandas as pd
import librosa
from deepface import DeepFace
import os
import tensorflow as tf
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel(logging.ERROR)

def extract_audio_features(video_path, fps):
    print("   🎧 Reading Audio...")
    y, sr = librosa.load(video_path, sr=None)
    hop_length = int(sr / fps)
    rms = librosa.feature.rms(y=y, frame_length=hop_length, hop_length=hop_length)[0]
    if rms.max() > 0: rms = rms / rms.max()
    return rms

def extract_all_features(video_path, output_csv=None):
    print(f"🕵️‍♀️ SPYING ON: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    features = []
    prev_gray = None
    
    # 1. First Pass: Extract Motion & Emotion
    print(f"   🎞️ Analyzing {frame_count} frames...")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        small_frame = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Motion
        motion_score = 0.0
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_score = np.sum(diff) / (640 * 360)
        prev_gray = gray

        # Emotion (Every 5th frame)
        prob_happy = 0.0
        prob_surprise = 0.0
        if frame_idx % 5 == 0:
            try:
                objs = DeepFace.analyze(img_path=small_frame, actions=['emotion'], 
                                        enforce_detection=False, silent=True, detector_backend='opencv')
                if objs:
                    emotions = objs[0]['emotion']
                    prob_happy = emotions.get('happy', 0) / 100.0
                    prob_surprise = emotions.get('surprise', 0) / 100.0
            except: pass
        elif features:
            prob_happy = features[-1]['prob_happy']
            prob_surprise = features[-1]['prob_surprise']

        features.append({
            'timestamp': frame_idx / fps,
            'motion_score': motion_score,
            'prob_happy': prob_happy,
            'prob_surprise': prob_surprise
        })
        frame_idx += 1

    cap.release()
    
    # 2. Add Audio
    df = pd.DataFrame(features)
    try:
        audio_rms = extract_audio_features(video_path, fps)
        if len(audio_rms) > len(df): audio_rms = audio_rms[:len(df)]
        else: audio_rms = np.pad(audio_rms, (0, len(df) - len(audio_rms)))
        df['rms_volume'] = audio_rms
    except: df['rms_volume'] = 0

    # --- 3. THE "HIGHLIGHT" LOGIC (Punch Detector) ---
    # A punch is defined as: HIGH Motion (> 0.5) AND HIGH Volume (> 0.4) at the same time.
    # We create a new column 'is_highlight'.
    
    print("   🥊 Calculating Action Highlights (Punch Logic)...")
    
    # You can tweak these thresholds. 
    # 0.5 motion is usually a fast swing. 0.4 volume is a loud impact.
    df['is_highlight'] = ((df['motion_score'] > 0.5) & (df['rms_volume'] > 0.4)).astype(int)
    
    # Optional: Expand the highlight (if a punch lasts 0.1s, we want to see 1s around it)
    # This "smears" the highlight to include the moments before/after the punch
    df['is_highlight'] = df['is_highlight'].rolling(window=int(fps), center=True, min_periods=1).max().fillna(0)

    # 4. Save
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"✅ Data Saved: {output_csv}")
        
    return df