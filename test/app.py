import streamlit as st
import cv2
import numpy as np
import librosa
import pandas as pd
import moviepy.editor as mp
import tempfile
import os
import requests
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SAMPLING_RATE = 0.5  # Extract data every 0.5 seconds
API_URL = "http://localhost:5000/predict" # Where Rohit's Brain lives

st.set_page_config(page_title="AI Video Editor", layout="wide")
st.title("🧠 AI Video Editor: Perception & Intelligence")

# --- STEP 1: UPLOAD VIDEO ---
uploaded_file = st.file_uploader("Upload Raw Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    col1, col2 = st.columns(2)
    with col1:
        st.video(video_path)
    with col2:
        st.info("Video Loaded. Ready to Extract Features.")

    # --- STEP 2: DATA MINING (LIBROSA + OPENCV) ---
    if st.button("🚀 Analyze Video & Ask AI"):
        st.write("---")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # A. Load Audio
        status_text.text("🎧 Extracting Audio Signal...")
        try:
            video_clip = mp.VideoFileClip(video_path)
            duration = video_clip.duration
            fps = video_clip.fps
            
            has_audio = video_clip.audio is not None
            audio_path = "temp_audio.wav"
            y = None
            sr = 22050
            
            if has_audio:
                video_clip.audio.write_audiofile(audio_path, logger=None)
                y, sr = librosa.load(audio_path, sr=sr)
            
        except Exception as e:
            st.error(f"Error processing video: {e}")
            st.stop()

        # B. The Loop
        status_text.text("👁 Analyzing Visuals & Audio...")
        timestamps = np.arange(0, duration, SAMPLING_RATE)
        data_records = []
        
        cap = cv2.VideoCapture(video_path)
        prev_gray = None
        
        for i, current_time in enumerate(timestamps):
            # Jump to frame
            frame_id = int(current_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            
            if not ret: break
            
            # --- Visual Features ---
            small_frame = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray) / 255.0
            
            motion = 0.0
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                motion = np.sum(diff) / (320 * 180 * 255)
            prev_gray = gray
            
            # --- Audio Features ---
            rms_val = 0.0
            zcr_val = 0.0
            
            if has_audio and y is not None:
                audio_idx = int(current_time * sr)
                window = int(SAMPLING_RATE * sr)
                start = max(0, audio_idx - window // 2)
                end = min(len(y), audio_idx + window // 2)
                
                if end > start:
                    chunk = y[start:end]
                    rms_val = float(np.sqrt(np.mean(chunk**2)))
                    # ZCR is crucial for distinguishing speech from noise
                    zcr_val = float(np.mean(librosa.feature.zero_crossing_rate(chunk)))

            data_records.append({
                'timestamp': round(current_time, 2),
                'rms_volume': round(rms_val, 4),
                'zcr_pitch': round(zcr_val, 4),
                'motion_score': round(motion, 4),
                'brightness': round(brightness, 3)
            })
            
            progress_bar.progress(min(i / len(timestamps), 1.0))
            
        cap.release()
        if has_audio and os.path.exists(audio_path):
            os.remove(audio_path)
            
        df = pd.DataFrame(data_records)
        status_text.text("✅ Feature Extraction Complete!")
        
        # --- STEP 3: VISUALIZATION (GRAPHS) ---
        st.subheader("📈 The 'Video DNA' Graph")
        st.caption("This is what the computer 'sees'. Peaks usually mean interesting moments.")
        
        # We use Matplotlib/Altair logic for a dual-axis chart
        chart_data = df.set_index('timestamp')[['rms_volume', 'motion_score']]
        st.line_chart(chart_data)
        
        # --- STEP 4: API CONNECTION (THE INTELLIGENCE) ---
        st.subheader("🤖 AI Decision Making")
        
        try:
            # Convert DF to JSON for the API
            payload = df.to_dict(orient='records')
            
            with st.spinner("Asking Rohit's Model for editing decisions..."):
                response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                decisions = response.json() # Returns [0, 1, 1, 0...]
                df['ai_decision'] = decisions
                
                # Show results
                st.success("Success! The AI has labeled the video.")
                
                # Color code the table: 1 = Green (Keep), 0 = Red (Cut)
                def highlight_decision(val):
                    color = 'green' if val == 1 else 'red'
                    return f'color: {color}'

                st.dataframe(df.style.applymap(highlight_decision, subset=['ai_decision']))
                
                # Summary Metric
                keep_count = df['ai_decision'].sum()
                total_count = len(df)
                st.metric("Retention Rate", f"{keep_count}/{total_count} Segments Kept")
                
            else:
                st.error(f"API Error: {response.status_code}")
                
        except Exception as e:
            st.warning(f"⚠️ Could not connect to API at {API_URL}. Is `ml_api.py` running?")
            st.info("For now, showing raw data only.")
            st.dataframe(df)
