import streamlit as st
import cv2
import numpy as np
import librosa
import pandas as pd
import moviepy.editor as mp
import tempfile
import os
import matplotlib.pyplot as plt
import editor_engine # Ensure editor_engine.py is in the folder

# --- CONFIGURATION ---
SAMPLING_RATE = 0.5  # Extract data every 0.5 seconds

st.set_page_config(page_title="AI Video Editor", layout="wide")
st.title("🧠 AI Video Editor: The Perception Engine (Standalone)")

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
        st.caption("Using Local Logic (Volume > 0.05 or Motion > 0.1)")

    # --- STEP 2: ANALYZE ---
    if st.button("🚀 Run Analysis & Edit"):
        st.write("---")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # --- A. AUDIO EXTRACTION (LIBROSA) ---
        status_text.text("🎧 Separating audio track...")
        
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
            else:
                y = np.zeros(int(duration * 22050))
                
        except Exception as e:
            st.error(f"Error processing video: {e}")
            st.stop()

        # --- B. FEATURE MINING LOOP ---
        status_text.text("👁 Analyzing visual & audio signals...")
        
        timestamps = np.arange(0, duration, SAMPLING_RATE)
        data_records = []
        
        cap = cv2.VideoCapture(video_path)
        prev_gray = None
        
        for i, current_time in enumerate(timestamps):
            frame_id = int(current_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret: break
            
            # Visuals
            small_frame = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray) / 255.0
            
            motion = 0.0
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                motion = np.sum(diff) / (320 * 180 * 255)
            prev_gray = gray
            
            # Audio
            rms_val = 0.0
            zcr_val = 0.0
            
            if has_audio:
                audio_idx = int(current_time * sr)
                window = int(SAMPLING_RATE * sr)
                start = max(0, audio_idx - window // 2)
                end = min(len(y), audio_idx + window // 2)
                
                if end > start:
                    chunk = y[start:end]
                    rms_val = float(np.sqrt(np.mean(chunk**2)))
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
        status_text.text("✅ Analysis Complete!")
        
        # --- STEP 3: VISUALIZATION (COMPACT) ---
        st.subheader("📊 The 'Video DNA' Dashboard")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3), sharex=True)
        
        librosa.display.waveshow(y, sr=sr, ax=ax1, alpha=0.5, color='blue')
        ax1.set_title("Audio", fontsize=8)
        ax1.tick_params(labelsize=6)
        
        ax2.plot(df['timestamp'], df['rms_volume'], color='green', linewidth=1)
        ax2.plot(df['timestamp'], df['motion_score'], color='orange', linewidth=1)
        ax2.set_title("AI Signals", fontsize=8)
        ax2.tick_params(labelsize=6)
        
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.pyplot(fig, use_container_width=True)

        # --- STEP 4: DECISION LOGIC (LOCAL BRAIN) ---
        st.subheader("🤖 AI Editor Decisions")
        
        def local_decision_logic(row):
            # Keep if loud enough (> 0.05) OR moving enough (> 0.1)
            if row['rms_volume'] > 0.05 or row['motion_score'] > 0.1:
                return 1 
            else:
                return 0 
                
        df['ai_decision'] = df.apply(local_decision_logic, axis=1)
        
        # Metric Summary
        keep_percent = int((df['ai_decision'].sum() / len(df)) * 100)
        st.metric("Retention Score", f"{keep_percent}%", "of video kept")

        # --- TABLE IS BACK HERE ---
        # def highlight_rows(row):
            # Green if Keep (1), Red if Cut (0)
            # color = '#d4edda' if row['ai_decision'] == 1 else '#f8d7da'
            # return [f'background-color: {color}' for _ in row]
        
        # st.dataframe(df.style.apply(highlight_rows, axis=1))
        # NEW CODE (Plain - No Colors)
        st.dataframe(df)

        # --- STEP 5: RENDER VIDEO ---
        st.write("---")
        st.write("### 🎬 Final Production")
        
        if st.button("✂️ Render Final Movie"):
            with st.spinner("The Director is editing..."):
                
                # CALL THE SEPARATE ENGINE
                output_file = editor_engine.process_and_render(video_path, df)
                
                if output_file:
                    st.success("✨ Render Complete!")
                    st.video(output_file)
                else:
                    st.error("The AI decided to cut the entire video.")
