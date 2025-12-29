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
import editor_engine

# --- CONFIGURATION ---
SAMPLING_RATE = 0.5  # Extract data every 0.5 seconds
API_URL = "http://localhost:5000/predict" # Link to your 'ml_api.py'

st.set_page_config(page_title="AI Video Editor", layout="wide")
st.title("🧠 AI Video Editor: The Perception Engine")

# --- STEP 1: UPLOAD VIDEO ---
uploaded_file = st.file_uploader("Upload Raw Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.video(video_path)
    with col2:
        st.info("Video Loaded. Ready for Deep Analysis.")
        st.write("This engine will extract visual motion and audio textures to feed the Neural Network.")

    # --- STEP 2: ANALYZE ---
    if st.button("🚀 Run Analysis & Visualization"):
        st.write("---")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # --- A. AUDIO EXTRACTION (LIBROSA) ---
        status_text.text("🎧 separating audio track...")
        
        try:
            video_clip = mp.VideoFileClip(video_path)
            duration = video_clip.duration
            fps = video_clip.fps
            
            # Extract raw audio for Librosa
            has_audio = video_clip.audio is not None
            audio_path = "temp_audio.wav"
            y = None
            sr = 22050
            
            if has_audio:
                video_clip.audio.write_audiofile(audio_path, logger=None)
                # Load into Librosa (y = audio array, sr = sample rate)
                y, sr = librosa.load(audio_path, sr=sr)
            else:
                st.warning("⚠️ No Audio Detected. Filling with silence.")
                y = np.zeros(int(duration * 22050))
                
        except Exception as e:
            st.error(f"Error processing video: {e}")
            st.stop()

        # --- B. FEATURE MINING LOOP ---
        status_text.text("👁 Analyzing frame-by-frame signals...")
        
        timestamps = np.arange(0, duration, SAMPLING_RATE)
        data_records = []
        
        cap = cv2.VideoCapture(video_path)
        prev_gray = None
        
        for i, current_time in enumerate(timestamps):
            # 1. Video Frame Processing
            frame_id = int(current_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret: break
            
            # Resize & Grayscale for speed
            small_frame = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray) / 255.0
            
            # Motion Calculation
            motion = 0.0
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                motion = np.sum(diff) / (320 * 180 * 255)
            prev_gray = gray
            
            # 2. Audio Chunk Processing
            rms_val = 0.0
            zcr_val = 0.0
            
            if has_audio:
                audio_idx = int(current_time * sr)
                window = int(SAMPLING_RATE * sr)
                start = max(0, audio_idx - window // 2)
                end = min(len(y), audio_idx + window // 2)
                
                if end > start:
                    chunk = y[start:end]
                    # RMS = Volume (Energy)
                    rms_val = float(np.sqrt(np.mean(chunk**2)))
                    # ZCR = Pitch/Noise Texture
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
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")
            
        df = pd.DataFrame(data_records)
        status_text.text("✅ Signal Processing Complete!")
        
        # --- STEP 3: PROFESSIONAL VISUALIZATION (MATPLOTLIB) ---
        # st.subheader("📊 The 'Video DNA' Dashboard")
        
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        #
        # # Plot 1: The Raw Audio Waveform (What Humans Hear)
        # # We downsample the audio by 100x just to make plotting faster
        # librosa.display.waveshow(y, sr=sr, ax=ax1, alpha=0.5, color='blue')
        # ax1.set_title("Raw Audio Waveform (Human Hearing)", fontsize=10)
        # ax1.set_ylabel("Amplitude")
        #
        # # Plot 2: The Extracted Features (What AI Sees)
        # ax2.plot(df['timestamp'], df['rms_volume'], label='Volume (RMS)', color='green', linewidth=2)
        # ax2.plot(df['timestamp'], df['motion_score'], label='Visual Motion', color='orange', linewidth=2)
        # ax2.set_title("Extracted AI Signals (Computer Vision)", fontsize=10)
        # ax2.set_xlabel("Time (seconds)")
        # ax2.set_ylabel("Intensity (0-1)")
        # ax2.legend()
        # ax2.grid(True, alpha=0.3)
        #
        # # Send Plot to Streamlit
        # st.pyplot(fig)

# --- STEP 3: PROFESSIONAL VISUALIZATION (COMPACT) ---
        st.subheader("📊 The 'Video DNA' Dashboard")
        
        # 1. Create a SMALL figure (Width=6, Height=3)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3), sharex=True)
        
        # Plot 1: Waveform
        librosa.display.waveshow(y, sr=sr, ax=ax1, alpha=0.5, color='blue')
        ax1.set_title("Audio", fontsize=8)
        ax1.tick_params(labelsize=6) # Make text smaller
        
        # Plot 2: Features
        ax2.plot(df['timestamp'], df['rms_volume'], color='green', linewidth=1)
        ax2.plot(df['timestamp'], df['motion_score'], color='orange', linewidth=1)
        ax2.set_title("AI Signals", fontsize=8)
        ax2.tick_params(labelsize=6)
        ax2.grid(True, alpha=0.3)
        
        # 2. THE TRICK: Use Columns to squeeze the image
        # We create 3 columns: [Empty space, Graph, Empty space]
        col_left, col_center, col_right = st.columns([1, 2, 1])

        with col_center:
            # We put the graph ONLY in the middle column
            st.pyplot(fig, use_container_width=True)


        # --- STEP 4: ASK THE API ---
        # st.subheader("🤖 AI Editor Decisions")
        
        try:
            payload = df.to_dict(orient='records')
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                df['ai_decision'] = response.json()
                
                # Metric Summary
                keep_percent = int((df['ai_decision'].sum() / len(df)) * 100)
                st.metric("AI Retention Score", f"{keep_percent}%", "of video kept")
                
                # Show the Data
                def highlight_rows(row):
                    color = '#d4edda' if row['ai_decision'] == 1 else '#f8d7da'
                    return [f'background-color: {color}' for _ in row]
                
                st.dataframe(df.style.apply(highlight_rows, axis=1))
                
            else:
                st.error("API returned an error.")
                
        except Exception:
            st.warning("Could not connect to AI Brain (ml_api.py). Showing Raw Signals only.")
            st.dataframe(df)

            if response.status_code == 200:
                df['ai_decision'] = response.json()
                st.success("AI has labeled the video.")

                # --- THE ACTION BUTTON ---
                if st.button("✂️ Create Final Movie"):
                    with st.spinner("The Director is editing..."):
                        
                        # CALL THE SEPARATE ENGINE
                        output_file = editor_engine.process_and_render(video_path, df)
                        
                        if output_file:
                            st.success("✨ Render Complete!")
                            st.video(output_file)
                        else:
                            st.error("The AI decided to cut the entire video.")
