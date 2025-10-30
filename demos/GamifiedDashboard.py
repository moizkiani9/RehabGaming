import streamlit as st
import pandas as pd
import cv2
import numpy as np
import time
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import internal modules
from modules.ArmRaise import ArmRaiseExercise
from modules.ExerciseAnalytics import ExerciseAnalytics


# ---------- Streamlit Config ----------
st.set_page_config(page_title="RehabGaming Dashboard", layout="wide", page_icon="🎮")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🎮 RehabGaming - Gamified Physiotherapy Dashboard</h1></div>', 
            unsafe_allow_html=True)

# Initialize session state
if "exercise_data" not in st.session_state:
    st.session_state.exercise_data = []
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "analytics" not in st.session_state:
    st.session_state.analytics = ExerciseAnalytics()
if "exercise_instance" not in st.session_state:
    st.session_state.exercise_instance = None
if "session_start_time" not in st.session_state:
    st.session_state.session_start_time = None


# ---------- Sidebar Controls ----------
st.sidebar.header("🧩 Exercise Controls")
exercise_type = st.sidebar.selectbox("Select Exercise", ["Arm Raise", "Knee Bend", "Shoulder Roll"])
st.sidebar.markdown("---")

# Control buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    start_button = st.button("▶ Start", use_container_width=True, type="primary")
with col2:
    stop_button = st.button("⏹ Stop", use_container_width=True)

reset_button = st.sidebar.button("🔄 Reset Session", use_container_width=True)

# Settings
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")
show_debug = st.sidebar.checkbox("Show Debug Info", value=False)
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.5, 1.0, 0.7, 0.05)

# Stats in sidebar
if st.session_state.exercise_instance:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Current Session")
    st.sidebar.metric("Reps", st.session_state.exercise_instance.rep_count)
    st.sidebar.metric("Points", st.session_state.exercise_instance.points)
    
    if st.session_state.session_start_time:
        elapsed = time.time() - st.session_state.session_start_time
        st.sidebar.metric("Time", f"{int(elapsed)}s")


# ---------- Main Exercise Area ----------
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"🎯 {exercise_type}")
    FRAME_WINDOW = st.empty()
    status_placeholder = st.empty()

with col_right:
    st.subheader("📈 Live Stats")
    stats_placeholder = st.empty()


# ---------- Button Actions ----------
if start_button and not st.session_state.is_running:
    st.session_state.is_running = True
    st.session_state.session_start_time = time.time()
    st.session_state.exercise_instance = ArmRaiseExercise(
        detection_confidence=confidence_threshold,
        tracking_confidence=confidence_threshold
    )
    st.rerun()

if stop_button and st.session_state.is_running:
    st.session_state.is_running = False
    
    # Save session data
    if st.session_state.exercise_instance:
        exercise = st.session_state.exercise_instance
        duration = time.time() - st.session_state.session_start_time if st.session_state.session_start_time else 0
        
        # Calculate average form score
        total_quality_reps = (
            exercise.perfect_reps + 
            exercise.good_reps + 
            exercise.okay_reps
        )
        
        if total_quality_reps > 0:
            avg_form = (
                (exercise.perfect_reps * 10 + 
                 exercise.good_reps * 7 + 
                 exercise.okay_reps * 5)
            ) / total_quality_reps
        else:
            avg_form = 0.0
        
        session_data = {
            "timestamp": datetime.now(),
            "exercise_type": exercise_type,
            "reps": exercise.rep_count,
            "avg_form_score": avg_form,
            "duration": duration
        }
        
        st.session_state.analytics.add_exercise_session(session_data)
        st.session_state.exercise_data.append(session_data)
    
    st.rerun()

if reset_button:
    st.session_state.exercise_data = []
    st.session_state.is_running = False
    st.session_state.exercise_instance = None
    st.session_state.session_start_time = None
    st.session_state.analytics = ExerciseAnalytics()  # Reset analytics too
    st.success("✅ Session reset successfully!")
    st.rerun()


# ---------- Live Exercise Logic ----------
if st.session_state.is_running:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        status_placeholder.error("⚠️ Camera not accessible. Please check your webcam.")
        st.session_state.is_running = False
        st.stop()
    
    status_placeholder.info("📸 Camera Active — Exercise in progress...")
    
    exercise = st.session_state.exercise_instance
    frame_count = 0
    
    # Create placeholders for stats
    with stats_placeholder.container():
        stat_col1, stat_col2 = st.columns(2)
        rep_metric = stat_col1.empty()
        point_metric = stat_col2.empty()
        
        stat_col3, stat_col4, stat_col5 = st.columns(3)
        perfect_metric = stat_col3.empty()
        good_metric = stat_col4.empty()
        okay_metric = stat_col5.empty()
        
        feedback_box = st.empty()
    
    try:
        while st.session_state.is_running:
            success, frame = cap.read()
            if not success:
                status_placeholder.error("⚠️ Failed to read frame from camera.")
                break
            
            # Mirror and process frame
            frame = cv2.flip(frame, 1)
            frame = exercise.process_frame(frame)
            
            # Convert for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Update stats every 10 frames for performance
            if frame_count % 10 == 0:
                rep_metric.metric("🏋️ Reps", exercise.rep_count)
                point_metric.metric("⭐ Points", exercise.points)
                perfect_metric.metric("Perfect", exercise.perfect_reps, 
                                    delta=f"+{exercise.perfect_reps*10} pts")
                good_metric.metric("Good", exercise.good_reps, 
                                 delta=f"+{exercise.good_reps*7} pts")
                okay_metric.metric("Okay", exercise.okay_reps, 
                                 delta=f"+{exercise.okay_reps*5} pts")
                
                # Show feedback
                if exercise.feedback:
                    feedback_box.info(f"💬 **{exercise.feedback}**")
            
            frame_count += 1
            
            # Small delay
            time.sleep(0.03)  # ~30 FPS
            
    except Exception as e:
        status_placeholder.error(f"⚠️ Error during exercise: {str(e)}")
    
    finally:
        cap.release()
        st.session_state.is_running = False

else:
    # Show placeholder when not running
    FRAME_WINDOW.info("👆 Click **Start** to begin your exercise session")
    stats_placeholder.info("📊 Stats will appear here during exercise")


# ---------- Analytics Display ----------
st.divider()
st.header("📊 Exercise Progress Analytics")

if len(st.session_state.exercise_data) > 0:
    metrics = st.session_state.analytics.get_progress_metrics(exercise_type)
    
    if metrics:
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏋️ Total Reps", metrics["total_reps"])
        col2.metric("⭐ Avg Form Score", f"{metrics['avg_form_score']:.1f}/10")
        col3.metric("🕒 Avg Duration", f"{metrics['avg_duration']:.1f}s")
        col4.metric("📈 Total Sessions", metrics["total_sessions"])
        
        # Weekly stats
        if 'weekly_stats' in metrics:
            st.subheader("📅 This Week's Progress")
            week_col1, week_col2, week_col3 = st.columns(3)
            week_col1.metric("Sessions", metrics['weekly_stats']['sessions_this_week'])
            week_col2.metric("Reps", metrics['weekly_stats']['reps_this_week'])
            week_col3.metric("Avg Form", f"{metrics['weekly_stats']['avg_form_this_week']:.1f}/10")
        
        # Improvement areas
        st.subheader("🎯 Areas to Improve")
        if metrics["improvement_areas"]:
            for area in metrics["improvement_areas"]:
                st.warning(f"⚠️ {area}")
        else:
            st.success("✅ Excellent consistency and form! Keep up the great work!")
        
        # Performance graph
        st.subheader("📈 Performance Over Time")
        if len(st.session_state.exercise_data) > 1:
            df = pd.DataFrame(st.session_state.exercise_data)
            
            # Create chart data
            chart_data = pd.DataFrame({
                'Reps': df['reps'].values,
                'Form Score': df['avg_form_score'].values,
            }, index=range(1, len(df) + 1))
            
            st.line_chart(chart_data)
        
        # Best session
        st.subheader("🏆 Best Session")
        best = metrics["best_session"]
        st.success(
            f"🗓 **Date:** {best['date']} | "
            f"💪 **Reps:** {best['reps']} | "
            f"⭐ **Form Score:** {best['form_score']:.1f}/10 | "
            f"⏱ **Duration:** {best['duration']:.0f}s"
        )
        
        # Session history table
        with st.expander("📋 View Session History"):
            df = pd.DataFrame(st.session_state.exercise_data)
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(df, use_container_width=True)
        
        # Export option
        if st.button("💾 Export Session Data"):
            export_df = pd.DataFrame(st.session_state.exercise_data)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"rehab_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

else:
    st.info("👆 Start an exercise session to see your analytics and progress!")
    st.markdown("""
    ### 📝 How to Use:
    1. Select an exercise from the sidebar
    2. Click **Start** to begin your session
    3. Perform the exercise following the on-screen guidance
    4. Click **Stop** when finished
    5. View your progress and analytics below
    """)


# ---------- Footer ----------
st.markdown("---")
st.caption("© 2025 RehabGaming | Built with ❤️ by Abdul Moiz Kiyani & Team | Powered by MediaPipe & Streamlit")