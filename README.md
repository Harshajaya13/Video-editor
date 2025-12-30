🎬 AI Video Editor (The "Smart Cutter")

An automated video editing pipeline that uses Computer Vision and Audio Analysis to "watch" your footage and autonomously create highlight reels. It separates the Analysis (The Eye), Decision Making (The Brain), and Rendering (The Scissors) into a distributed workflow, allowing for remote processing of heavy logic.
🧠 How It Works

The application operates on a Extract -> Refine -> Render philosophy, mimicking a professional human editor's workflow:

    The Eye (Feature Extraction): * Scans the video frame-by-frame.

        Visuals: Calculates Motion Score (optical flow), Brightness, and detects Face Emotions (Happy/Sad/Surprise) using DeepFace.

        Audio: Extracts RMS Volume and Pitch (Zero Crossing Rate) using Librosa.

        Output: A raw features.csv.

    The Brain (Decision Logic): * Can run locally or as a remote Flask API.

        Normalizes data and applies weighted scoring based on user preference (e.g., "Make it 80% Action, 20% Emotion").

        Uses a "Sort & Slice" algorithm to fit the best moments into a specific target duration (e.g., 30 seconds).

        Output: A refined.csv with binary cut decisions.

    The Scissors (Rendering Engine): * Takes the original video and the refined.csv.

        Uses MoviePy to physically cut, rearrange, and stitch the clips.

        Applies smart audio fades (smoother fades for loud clips, sharp cuts for quiet ones).

🛠️ Installation
Prerequisites

    Python 3.10 (Required for TensorFlow/DeepFace compatibility).

    FFmpeg installed on your system.

1. Clone the Repository
Bash

git clone https://github.com/yourusername/ai-video-editor.git
cd ai-video-editor

2. Install Python Dependencies
Bash

pip install -r requirements.txt

3. Install System Packages (Linux/Cloud Only)

If deploying to Streamlit Cloud or Linux, ensure packages.txt is read, or manually install:
Bash

sudo apt-get install ffmpeg libgl1

🚀 Usage Guide

To start the User Interface:
Bash

streamlit run app.py

Phase 1: Feature Extraction

    Go to Tab 1.

    Upload your raw video file (.mp4).

    Click "Start Feature Extraction".

    Download the generated features.csv.

Phase 2: The Decision (Distributed Workflow)

You can process the CSV locally in Tab 2 OR send it to a remote server.

Option A: Remote API (Postman)

    Send features.csv via POST request to the Logic Server.

    Receive refined.csv in response.

Option B: In-App Logic

    Go to Tab 2.

    Upload the raw video AND the features.csv.

    Adjust the sliders (Action, Happiness, Loudness).

    Set target duration (e.g., 30s).

Phase 3: Final Render

    If you used Option A, upload the refined.csv back into Tab 2.

    The app detects the refinement and locks the sliders.

    Click "✨ Render Final Video".

    Watch or download your AI-edited clip.

📡 API / Server Mode (Optional)

If you are running the "Brain" on a separate machine (e.g., a friend's laptop), run the Flask server script:
Bash

python server_brain.py

API Endpoint: POST /refine

    Body (form-data):

        file: The features.csv file.

        seconds: (Int) Target duration in seconds.

⚙️ Configuration & Tuning

You can tweak the weights in the Edit Settings expander:

    Motion: High value prioritizes fast movement (sports, running).

    Loudness: High value prioritizes shouting, cheering, or loud music.

    Happiness: Prioritizes smiling faces.

    Sadness: Prioritizes somber expressions.

"Punch Logic": The app automatically detects "Highlights" (High Motion + High Audio Impact) and forces them to be included if the "Keep Highlights" checkbox is ticked.
📦 Deployment (Streamlit Cloud)

This app is optimized for Streamlit Community Cloud.

Critical requirements.txt Setup: Ensure your requirements file is pinned exactly as below to avoid Version Conflicts between Keras 3 and DeepFace:
Plaintext

streamlit
pandas
numpy
moviepy==1.0.3
librosa
opencv-python-headless
tensorflow-cpu==2.15.0
keras==2.15.0
deepface
tf-keras

🐛 Troubleshooting

    "MoviePy error" / Import Error: Ensure you are using moviepy==1.0.3. Version 2.0+ breaks the editor engine.

    "Visual C++ / DLL Load Failed": Install the Visual C++ Redistributable on Windows.

    "Out of Memory": DeepFace is heavy. If running on a free cloud tier, try reducing the video resolution before upload or using tensorflow-cpu.

📝 License

This project is open-source. Feel free to fork and modify!
