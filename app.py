import streamlit as st
import pandas as pd
import tempfile
import os
import editor_engine      # The Scissors (Cuts the video)
import feature_extractor  # The Eyes (Extracts volume, emotions, brightness)

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
    st.markdown("This step scans the video for **Volume**, **Motion**, and **Emotions**.")
    
    if st.session_state['video_path']:
        if st.button("🚀 Start Feature Extraction"):
            with st.spinner("Analyzing frames for smiles, audio, and action..."):
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

    # Allow downloading the raw data for Rohit (or yourself)
    if st.session_state['df_features'] is not None:
        csv = st.session_state['df_features'].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Raw CSV", csv, "raw_features.csv", "text/csv")
        
 # ==========================================
# TAB 2: AI EDITING (The Brain & Scissors)
# ==========================================
with tab2:
    st.header("🎬 Step 2: Create the Edit")
    
    # 1. UPLOAD BOX (Works for BOTH Raw files and Rohit's files)
    uploaded_csv = st.file_uploader("Upload CSV (Raw from Tab 1 OR Scored from Rohit)", type=["csv"])
    
    # Determine which DataFrame to use
    active_df = None
    if uploaded_csv:
        active_df = pd.read_csv(uploaded_csv)
    elif st.session_state['df_features'] is not None:
        active_df = st.session_state['df_features'].copy()
        
    if active_df is not None:
        st.divider()
        
        # --- SMART LOGIC: Is this Rohit's file or a Raw file? ---
        
        # Check if Rohit already gave us a score (looking for 'score', 'pred', or 'engagement_score')
        # We normalize everything to 'engagement_score'
        is_pre_scored = False
        
        if 'score' in active_df.columns:
            active_df['engagement_score'] = active_df['score']
            is_pre_scored = True
        elif 'pred' in active_df.columns:
            active_df['engagement_score'] = active_df['pred']
            is_pre_scored = True
        elif 'engagement_score' in active_df.columns:
            is_pre_scored = True

        # --- PATH A: ROHIT'S FILE (Pre-Scored) ---
        if is_pre_scored:
            st.success("✨ Rohit's Pre-Scored Data Detected! Skipping manual calculation.")
            st.line_chart(active_df['engagement_score'])
            
        # --- PATH B: RAW FILE (Manual Calculation) ---
        else:
            st.subheader("🧠 The Brain: Define 'Engagement'")
            st.info("No score found. Using Manual Sliders.")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: w_vol = st.slider("🔊 Volume Weight", 0.0, 5.0, 1.0)
            with col2: w_happy = st.slider("😊 Happy Weight", 0.0, 5.0, 2.0)
            with col3: w_shock = st.slider("😲 Surprise Weight", 0.0, 5.0, 2.5)
            with col4: w_motion = st.slider("🏃 Motion Weight", 0.0, 5.0, 1.0)

            # Normalize Volume
            max_vol = active_df['rms_volume'].max()
            if max_vol == 0: max_vol = 1
            active_df['norm_vol'] = active_df['rms_volume'] / max_vol

            # Safety checks for missing columns
            if 'prob_happy' not in active_df: active_df['prob_happy'] = 0
            if 'prob_surprise' not in active_df: active_df['prob_surprise'] = 0
            if 'motion_score' not in active_df: active_df['motion_score'] = 0

            # CALCULATE SCORE MANUALLY
            active_df['engagement_score'] = (
                (active_df['norm_vol'] * w_vol) +
                (active_df['prob_happy'] * w_happy) +
                (active_df['prob_surprise'] * w_shock) +
                (active_df['motion_score'] * w_motion)
            )
            st.line_chart(active_df['engagement_score'])

        # --- B. EDITING CONTROLS (The Scissors) ---
        # This part runs for BOTH paths!
        st.divider()
        st.subheader("✂️ The Cut")
        
        strictness = st.slider("Strictness (Keep Top %)", 0.1, 1.0, 0.4)
        
        # Calculate Threshold
        threshold = active_df['engagement_score'].quantile(1.0 - strictness)
        st.write(f"**Keeping segments with Score > {threshold:.2f}**")
        
        # APPLY DECISION (Create the 'ai_decision' column)
        active_df['ai_decision'] = active_df['engagement_score'].apply(lambda x: 1 if x >= threshold else 0)
        
        keep_count = active_df['ai_decision'].sum()
        st.caption(f"Will keep {keep_count} seconds of video.")

        if st.button("✨ Render Final Video"):
            if st.session_state['video_path']:
                with st.spinner("Cutting and stitching..."):
                    output_file = editor_engine.process_and_render(
                        st.session_state['video_path'], 
                        active_df
                    )
                    if output_file:
                        st.success("🎉 Done!")
                        st.video(output_file)
            else:
                st.error("Video file is missing.")
    else:
        st.info("Waiting for data... Go to Tab 1 to analyze a video OR upload a CSV here.")