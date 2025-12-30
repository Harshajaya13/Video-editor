import cv2
import numpy as np
import pandas as pd
import librosa
from deepface import DeepFace
import os
import tensorflow as tf
import logging

# Mute TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel(logging.ERROR)

def extract_audio_features(video_path, fps):
    print("   🎧 Reading Audio...")
    y, sr = librosa.load(video_path, sr=None)
    hop_length = int(sr / fps)
    
    # 1. Volume (RMS)
    rms = librosa.feature.rms(y=y, frame_length=hop_length, hop_length=hop_length)[0]
    if rms.max() > 0: rms = rms / rms.max()
    
    # 2. Pitch/Impact (Zero Crossing Rate)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=hop_length, hop_length=hop_length)[0]
    if zcr.max() > 0: zcr = zcr / zcr.max()
        
    return rms, zcr

def extract_all_features(video_path, output_csv=None):
    print(f"🕵️‍♀️ SPYING ON: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    features = []
    prev_gray = None
    
    print(f"   🎞️ Analyzing {frame_count} frames...")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        small_frame = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # --- 1. VISUAL METRICS ---
        
        # A. Motion Score
        motion_score = 0.0
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_score = np.sum(diff) / (640 * 360)
        prev_gray = gray
        
        # B. Brightness
        brightness = np.mean(gray)

        # C. Emotions (Happy, Sad, Surprise, Neutral)
        prob_happy = 0.0
        prob_surprise = 0.0
        prob_sad = 0.0
        prob_neutral = 0.0
        
        # Analyze every 5th frame for speed
        if frame_idx % 5 == 0:
            try:
                objs = DeepFace.analyze(img_path=small_frame, actions=['emotion'], 
                                        enforce_detection=False, silent=True, detector_backend='opencv')
                if objs:
                    emotions = objs[0]['emotion']
                    prob_happy = emotions.get('happy', 0) / 100.0
                    prob_surprise = emotions.get('surprise', 0) / 100.0
                    prob_sad = emotions.get('sad', 0) / 100.0
                    prob_neutral = emotions.get('neutral', 0) / 100.0
            except: pass
        elif features:
            # Fill gaps with previous frame data
            prob_happy = features[-1]['prob_happy']
            prob_surprise = features[-1]['prob_surprise']
            prob_sad = features[-1]['prob_sad']
            prob_neutral = features[-1]['prob_neutral']

        features.append({
            'timestamp': frame_idx / fps,
            'motion_score': motion_score,
            'brightness': brightness,
            'prob_happy': prob_happy,
            'prob_surprise': prob_surprise,
            'prob_sad': prob_sad,
            'prob_neutral': prob_neutral
        })
        frame_idx += 1

    cap.release()
    
    # --- 2. AUDIO METRICS ---
    df = pd.DataFrame(features)
    try:
        rms, zcr = extract_audio_features(video_path, fps)
        
        # Sync lengths
        min_len = min(len(df), len(rms))
        df = df.iloc[:min_len]
        df['rms_volume'] = rms[:min_len]
        df['zcr_pitch'] = zcr[:min_len]
        
    except Exception as e:
        print(f"⚠️ Audio Error: {e}")
        df['rms_volume'] = 0
        df['zcr_pitch'] = 0

    # --- 3. HIGHLIGHT LOGIC ---
    print("   🥊 Calculating Highlights...")
    
    # Normalize brightness for calculation
    norm_bright = df['brightness'] / 255.0
    
    # LOGIC: Significant Motion + (Loud OR Sharp Sound) + Visible Scene
    is_action = (df['motion_score'] > 0.2)
    is_loud = (df['rms_volume'] > 0.2) | (df['zcr_pitch'] > 0.2)
    is_visible = (norm_bright > 0.1)
    
    df['is_highlight'] = (is_action & is_loud & is_visible).astype(int)

    # Pad highlights
    df['is_highlight'] = df['is_highlight'].rolling(window=int(fps), center=True, min_periods=1).max().fillna(0)

    # Save
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"✅ Data Saved: {output_csv}")
        
    return df
