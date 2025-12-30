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
 # ... (Keep imports and Tab 1 as is) ...

# ==========================================
# TAB 2: AI EDITING
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
        
        # --- CHECK FOR HIGHLIGHTS ---
        has_highlights = 'is_highlight' in active_df.columns and active_df['is_highlight'].sum() > 0
        
        if has_highlights:
            st.success(f"🥊 Action Highlights Detected! ({int(active_df['is_highlight'].sum())} frames)")
            # Show a chart of where the punches are
            st.caption("Highlight Timeline (1 = Action Detected)")
            st.area_chart(active_df['is_highlight'], height=100, color="#ff4b4b")
        
        # --- SCORING LOGIC ---
        st.subheader("🧠 Editing Logic")
        
        # Check for Rohit's Score
        if 'score' in active_df.columns: 
            active_df['engagement_score'] = active_df['score']
            st.info("Using Pre-Calculated Scores.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1: w_vol = st.slider("🔊 Volume", 0.0, 5.0, 1.0)
            with col2: w_happy = st.slider("😊 Happy", 0.0, 5.0, 2.0)
            with col3: w_shock = st.slider("😲 Surprise", 0.0, 5.0, 2.5)
            with col4: w_motion = st.slider("🏃 Motion", 0.0, 5.0, 1.0)

            # Standard Calc
            active_df['norm_vol'] = active_df['rms_volume'] / active_df['rms_volume'].max()
            for col in ['prob_happy', 'prob_surprise', 'motion_score']:
                if col not in active_df: active_df[col] = 0

            active_df['engagement_score'] = (
                (active_df['norm_vol'] * w_vol) +
                (active_df['prob_happy'] * w_happy) +
                (active_df['prob_surprise'] * w_shock) +
                (active_df['motion_score'] * w_motion)
            )

        # --- THE CUT ---
        st.divider()
        col_cut1, col_cut2 = st.columns(2)
        with col_cut1:
            strictness = st.slider("Strictness (Keep Top %)", 0.1, 1.0, 0.4)
            threshold = active_df['engagement_score'].quantile(1.0 - strictness)
            
        with col_cut2:
            # THE MAGIC SWITCH
            force_highlights = st.checkbox("🔥 Always Keep Highlights?", value=True, 
                                          help="If checked, any 'Punch/Action' detected will be kept, even if the score is low.")

        # --- FINAL DECISION LOGIC ---
        def make_decision(row):
            # 1. If it's a Highlight AND we want to force keep it -> KEEP (1)
            if force_highlights and row.get('is_highlight', 0) == 1:
                return 1
            # 2. Otherwise, check the score against the threshold
            elif row['engagement_score'] >= threshold:
                return 1
            # 3. Otherwise, cut it
            else:
                return 0

        active_df['ai_decision'] = active_df.apply(make_decision, axis=1)
        
        st.caption(f"Will keep {active_df['ai_decision'].sum()} frames.")

        if st.button("✨ Render Final Video"):
            if st.session_state['video_path']:
                with st.spinner("Cutting (Prioritizing Highlights)..."):
                    output_file = editor_engine.process_and_render(
                        st.session_state['video_path'], 
                        active_df
                    )
                    if output_file:
                        st.success("🎉 Done!")
                        st.video(output_file)