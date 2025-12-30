import streamlit as st
import pandas as pd
import tempfile
import os
import editor_engine      # The Scissors
import feature_extractor  # The Eyes

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Video Editor", layout="wide", page_icon="🎬")
st.title("🧠 AI Video Editor: The Emotion Engine")

# --- INITIALIZE SESSION STATE ---
if 'video_path' not in st.session_state:
    st.session_state['video_path'] = None
if 'df_features' not in st.session_state:
    st.session_state['df_features'] = None

# --- SIDEBAR: GLOBAL SETTINGS ---
with st.sidebar:
    st.header("📁 Project Files")
    uploaded_video = st.file_uploader("Upload Raw Video (MP4/MOV)", type=["mp4", "mov", "avi"])
    
    if uploaded_video:
        # Save video to temp file only once
        if st.session_state['video_path'] is None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            st.session_state['video_path'] = tfile.name
        
        st.video(st.session_state['video_path'])
        st.success(f"Video Loaded: {uploaded_video.name}")

# --- MAIN TABS ---
tab1, tab2 = st.tabs(["1️⃣ Extract Features (The Eyes)", "2️⃣ AI Editor (The Brain)"])

# ==========================================
# TAB 1: FEATURE EXTRACTION (The Spy)
# ==========================================
with tab1:
    st.header("🕵️‍♀️ Step 1: Analyze Video")
    st.markdown("This step scans the video for **Volume**, **Motion**, **Brightness** and **Emotions**.")
    
    if st.session_state['video_path']:
        if st.button("🚀 Start Feature Extraction"):
            with st.spinner("Analyzing frames... (This uses AI, please wait)"):
                # Call the Feature Extractor Script
                df = feature_extractor.extract_all_features(
                    st.session_state['video_path'], 
                    output_csv="raw_features_for_rohit.csv"
                )
                st.session_state['df_features'] = df
                st.success("✅ Analysis Complete!")
                st.dataframe(df.head())
    else:
        st.info("Please upload a video in the sidebar first.")

    # Allow downloading the raw data
    if st.session_state['df_features'] is not None:
        csv = st.session_state['df_features'].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Raw CSV", csv, "raw_features.csv", "text/csv")

# ... (Keep imports and Tab 1 the same) ...
import cv2
import pandas as pd
import tempfile
import os

def process_and_render(video_path, df_decisions):
    # 1. Setup Input
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # GET EXACT INPUT FPS (Crucial for Speed Fix)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0 or input_fps > 120: input_fps = 30 # Fallback
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 2. Setup Output
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = tfile.name
    
    # Use 'mp4v' for broad compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, input_fps, (width, height))
    
    # 3. The Cutting Process
    # We convert the decision column to a list for fast access
    # Ensure the dataframe aligns with frames. 
    # If DF is shorter than video, we stop cutting when DF ends.
    decisions = df_decisions['ai_decision'].values
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Check if we should keep this frame
        if frame_idx < len(decisions):
            if decisions[frame_idx] == 1:
                out.write(frame)
        else:
            # If video is longer than analysis, stop or skip
            break
            
        frame_idx += 1

    cap.release()
    out.release()
    
    return output_path

# ==========================================
# TAB 2: AI EDITING (Bulletproof Brain)
# ==========================================
with tab2:
    st.header("🎬 Step 2: Create the Edit")
    
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])
    
    active_df = None
    if uploaded_csv:
        active_df = pd.read_csv(uploaded_csv)
    elif st.session_state['df_features'] is not None:
        active_df = st.session_state['df_features'].copy()
        
    if active_df is not None:
        st.divider()
        
        # --- 1. SAFEGUARD: Auto-Fill Missing Columns ---
        # This prevents crashes if you are using an old CSV
        required_cols = ['prob_happy', 'prob_sad', 'motion_score', 'rms_volume', 'zcr_pitch', 'brightness', 'is_highlight']
        for col in required_cols:
            if col not in active_df.columns:
                active_df[col] = 0.0 # Fill missing data with 0
        
        # Normalize
        if active_df['rms_volume'].max() > 0:
            active_df['norm_vol'] = active_df['rms_volume'] / active_df['rms_volume'].max()
        else:
            active_df['norm_vol'] = 0

        # Estimate FPS
        total_frames = len(active_df)
        total_duration = active_df['timestamp'].max()
        est_fps = total_frames / (total_duration if total_duration > 0 else 1)
        if est_fps < 1: est_fps = 30
        
        st.caption(f"🎞️ Source: {total_duration/60:.1f} mins | {int(total_frames)} frames")

        # --- 2. THE SLIDERS ---
        st.subheader("🧠 Logic Settings")
        c1, c2, c3, c4 = st.columns(4)
        with c1: w_vol = st.slider("🔊 Loudness", 0.0, 5.0, 1.0)
        with c2: w_action = st.slider("🏃 Motion", 0.0, 5.0, 1.5)
        with c3: w_happy = st.slider("😊 Happy", 0.0, 5.0, 1.0)
        with c4: w_sad = st.slider("😢 Sadness", 0.0, 5.0, 1.0)

        # Calculate Engagement Score
        active_df['engagement_score'] = (
            (active_df['norm_vol'] * w_vol) +
            (active_df['motion_score'] * w_action) +
            (active_df['prob_happy'] * w_happy) +
            (active_df['prob_sad'] * w_sad) +
            (active_df['zcr_pitch'] * 0.5) # Bonus for sharp sounds
        )

        # --- 3. TARGET DURATION (The Fix) ---
        st.divider()
        st.subheader("⏱️ How long should the video be?")
        
        col_t, col_check = st.columns([2, 1])
        with col_t:
            target_seconds = st.number_input("Target Duration (Seconds)", min_value=5, max_value=int(total_duration), value=30)
            
        with col_check:
            force_highlights = st.checkbox("🔥 Keep Highlights", value=True, help="Always keep punches/action even if score is low.")

        # --- 4. THE DECISION LOGIC (Sort & Slice) ---
        # This method is impossible to break.
        
        # A. Calculate how many frames we need
        target_frame_count = int(target_seconds * est_fps)
        
        # B. Sort all frames by score (Highest to Lowest)
        sorted_indices = active_df.sort_values('engagement_score', ascending=False).index
        
        # C. Pick the top N frames
        top_indices = sorted_indices[:target_frame_count]
        
        # D. Mark them as "Keep" (1)
        active_df['ai_decision'] = 0
        active_df.loc[top_indices, 'ai_decision'] = 1
        
        # E. Force Highlights (Optional)
        if force_highlights:
            active_df.loc[active_df['is_highlight'] == 1, 'ai_decision'] = 1
            
        # F. Smoothing (Fill small gaps)
        # 10 frame window (~0.3s) prevents choppy cuts
        active_df['ai_decision'] = active_df['ai_decision'].rolling(window=10, center=True, min_periods=1).max().fillna(0)

        # --- 5. STATS & RENDER ---
        final_frames = active_df['ai_decision'].sum()
        final_duration = final_frames / est_fps
        
        st.metric(
            "Expected Result", 
            f"{final_duration:.1f} Seconds", 
            delta=f"{final_duration - target_seconds:.1f}s variance"
        )
        
        if final_duration == 0:
            st.error("⚠️ No frames selected! Try increasing the Target Duration.")
        
        if st.button("✨ Render Video"):
            if st.session_state['video_path']:
                status = st.empty()
                status.info("🚀 Starting Render Engine...")
                
                try:
                    # CALL THE ENGINE
                    output_file = editor_engine.process_and_render(
                        st.session_state['video_path'], 
                        active_df
                    )
                    
                    if output_file:
                        status.success("🎉 Done!")
                        st.video(output_file)
                    else:
                        status.error("❌ Render Engine returned None. Check Terminal.")
                except Exception as e:
                    status.error(f"❌ Crash: {e}")