RehabGaming - Gamified Physiotherapy 

An interactive rehabilitation and physiotherapy platform that uses computer vision and gamification to make exercises engaging and trackable.

##  Features

- 🎯 Real-time pose detection using MediaPipe
- 🏋️ Exercise tracking (Arm Raise, Knee Bend, Shoulder Roll)
- 📊 Performance analytics and progress tracking
- 🎮 Gamified scoring system (Perfect/Good/Okay reps)
- 📈 Interactive dashboard with Streamlit
- 💾 Export session data to CSV

## Installation

1. Clone this repository:
\`\`\`bash
git clone https://github.com/YOUR_USERNAME/RehabGaming.git
cd RehabGaming
\`\`\`

2. Create a virtual environment:
\`\`\`bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## 📖 Usage

### Run the Streamlit Dashboard:
\`\`\`bash
streamlit run demos/GamifiedDashboard.py
\`\`\`

### Run Individual Demos:

# Pose detection demo
python demos/PoseDetectionDemo.py

# Arm raise exercise demo
python demos/ArmRaise_Webcam.py

# Analytics demo
python demos/AnalyticsDemo.py
\`\`\`



## 🛠️ Technologies Used

- **Python 3.8+**
- **OpenCV** - Computer vision
- **MediaPipe** - Pose estimation
- **Streamlit** - Web dashboard
- **Pandas** - Data analysis
- **NumPy** - Numerical computing

## 👥 Authors

- **Abdul Moiz Kiyani** & Team

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- MediaPipe by Google
- Streamlit Community
"@ | Out-File -FilePath README.md -Encoding UTF8


