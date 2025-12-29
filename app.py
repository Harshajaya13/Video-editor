import streamlit as st
import pandas as pd
import tempfile
import os
import editor_engine  # Ensure editor_engine.py is in the folder

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Video Editor", layout="wide")
st.title("🧠 AI Video Editor: The Perception Engine")

# --- INITIALIZE SESSION STATE ---
# This acts as the "Memory" of the app so data isn't lost when you click buttons
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'video_path' not in st.session_state:
    st.session_state['video_path'] = None
if 'analysis_done' not in st.session_state:
    st.session_state['analysis_done'] = False

# --- STEP 1: UPLOAD FILES ---
col1, col2 = st.columns(2)

with col1:
    uploaded_video = st.file_uploader("1️⃣ Upload Raw Video", type=["mp4", "mov", "avi"])

with col2:
    uploaded_csv = st.file_uploader("2️⃣ Upload AI Scores (CSV)", type=["csv"], help="Upload the scored_features.csv here")

# Handle Video Upload
if uploaded_video:
    # Only save if we haven't already (prevents reloading on every click)
    if st.session_state['video_path'] is None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        st.session_state['video_path'] = tfile.name
    
    st.video(st.session_state['video_path'])

# --- STEP 2: ANALYZE BUTTON ---
if st.button("🚀 Load & Analyze Data"):
    if uploaded_csv is not None:
        st.info("📂 Loading Engagement Scores from CSV...")
        try:
            # 1. READ CSV (Handle headers/no-headers)
            try:
                df = pd.read_csv(uploaded_csv)
                # Check if first column is numeric (implies no header)
                float(df.columns[0])
                uploaded_csv.seek(0)
                df = pd.read_csv(uploaded_csv, header=None)
            except:
                pass # Likely has headers

            # 2. MAP COLUMNS
            # Try to find 'score' and 'timestamp'
            if 'score' not in df.columns:
                # Assume Index 5 is score based on your data
                df['score'] = df.iloc[:, 5]
            if 'timestamp' not in df.columns:
                df['timestamp'] = df.iloc[:, 0]

            # 3. CLEAN DATA (Force Numeric)
            df['score'] = pd.to_numeric(df['score'], errors='coerce')
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df.dropna(subset=['score', 'timestamp'], inplace=True)

            # 4. ADD DUMMY VOLUME (Safety for Engine)
            if 'rms_volume' not in df.columns:
                df['rms_volume'] = 0.5

            # SAVE TO SESSION STATE
            st.session_state['df'] = df
            st.session_state['analysis_done'] = True
            st.success("✅ Data Loaded Successfully!")

        except Exception as e:
            st.error(f"Error processing CSV: {e}")
    else:
        st.warning("⚠️ Please upload the 'scored_features.csv' first.")

# --- STEP 3: INTERACTIVE EDITING (Visible only after analysis) ---
if st.session_state['analysis_done'] and st.session_state['df'] is not None:
    st.write("---")
    st.subheader("🎛️ Director's Cut Controls")
    
    df = st.session_state['df'].copy() # Work on a copy
    
    # 1. THE SLIDER (Dynamic Adjustment)
    # Users can drag this and immediately see the graph change
    strictness = st.slider("Strictness (Keep Top %)", 
                           min_value=0.1, max_value=1.0, value=0.4, 
                           help="Lower = Keep More Video. Higher = Keep Only Best Parts.")
    
    # 2. APPLY LOGIC INSTANTLY
    threshold = df['score'].quantile(1.0 - strictness)
    df['ai_decision'] = df['score'].apply(lambda x: 1 if x >= threshold else 0)
    
    # 3. SHOW STATS
    col_a, col_b = st.columns(2)
    keep_count = df['ai_decision'].sum()
    total_count = len(df)
    retention = int((keep_count / total_count) * 100)
    
    col_a.metric("Retention Rate", f"{retention}%", f"Keeping {keep_count} segments")
    col_a.write(f"**Score Cutoff:** {threshold:.4f}")
    
    # 4. VISUALIZE
    # Color the chart: Green for Keep, Red for Cut? 
    # Streamlit charts are simple, so we just show the score line with a threshold rule
    st.line_chart(df[['score']])
    st.caption("The line represents engagement. Everything above the threshold (implicit) is kept.")

    # --- STEP 4: RENDER BUTTON ---
    st.write("---")
    if st.button("✂️ Render Final Movie"):
        if st.session_state['video_path']:
            with st.spinner("🎬 The AI is cutting and stitching your video..."):
                
                # Pass the dynamically modified 'df' to the engine
                output_file = editor_engine.process_and_render(st.session_state['video_path'], df)
                
                if output_file:
                    st.success("✨ Render Complete! Here is your AI Edit:")
                    st.video(output_file)
                else:
                    st.error("❌ The engine returned no video. Try lowering the strictness.")
        else:
            st.error("❌ Video file lost. Please upload again.")
