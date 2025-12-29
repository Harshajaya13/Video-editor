import pandas as pd
import editor_engine
import os
import sys

# --- CONFIGURATION ---
VIDEO_FILE = "input_video.mp4"       
CSV_FILE = "scored_features.csv"     
KEEP_TOP_PERCENT = 0.40              # <--- NEW SETTING: Keep top 40% of best moments

# 1. LOAD DATA
if not os.path.exists(CSV_FILE):
    print("❌ CSV file not found!")
    sys.exit()

df = pd.read_csv(CSV_FILE, header=None)

# 2. MAP & CLEAN COLUMNS
try:
    df['timestamp'] = df.iloc[:, 0]
    df['score'] = df.iloc[:, 5]
except:
    # Fallback if headers exist
    df = pd.read_csv(CSV_FILE)
    if 'time' in df.columns: df['timestamp'] = df['time']
    # If score column has a different name, we might need to find it, 
    # but let's assume index 5 for now based on your data.
    if 'score' not in df.columns:
         # Try to grab the last column if named differently
         df['score'] = df.iloc[:, -1]

# Force numeric
df['score'] = pd.to_numeric(df['score'], errors='coerce')
df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
df.dropna(subset=['score', 'timestamp'], inplace=True)

# Add dummy volume
if 'rms_volume' not in df.columns:
    df['rms_volume'] = 0.5 

# --- 3. DYNAMIC THRESHOLD CALCULATION (THE FIX) ---
# See what the scores actually look like
min_score = df['score'].min()
max_score = df['score'].max()
avg_score = df['score'].mean()

print(f"📊 Score Stats -> Min: {min_score:.2f} | Max: {max_score:.2f} | Avg: {avg_score:.2f}")

# Calculate the cutoff score for the top X%
# quantile(0.6) means "the score that is higher than 60% of the rest" (Top 40%)
dynamic_threshold = df['score'].quantile(1.0 - KEEP_TOP_PERCENT)

print(f"⚖️  Dynamic Threshold: {dynamic_threshold:.4f} (Keeping Top {int(KEEP_TOP_PERCENT*100)}%)")

# Apply Logic
df['ai_decision'] = df['score'].apply(lambda x: 1 if x >= dynamic_threshold else 0)

kept_count = df['ai_decision'].sum()
total_count = len(df)
print(f"✨ Keeping {kept_count} / {total_count} segments")

# 4. RENDER
output_file = "final_ai_smart_edit.mp4"
editor_engine.process_and_render(VIDEO_FILE, df, output_path=output_file)
