import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import statistics


class ExerciseAnalytics:
    """
    Analytics engine for tracking and analyzing exercise performance.
    Provides metrics, insights, and progress tracking.
    """
    
    def __init__(self):
        """Initialize the analytics engine with empty session history."""
        self.sessions: List[Dict] = []
        self.session_history = pd.DataFrame(columns=[
            'timestamp', 'exercise_type', 'reps', 'avg_form_score', 'duration'
        ])
    
    def add_exercise_session(self, session_data: Dict):
        """
        Add a completed exercise session to the history.
        
        Args:
            session_data: Dictionary containing:
                - timestamp: datetime object
                - exercise_type: str (e.g., "Arm Raise")
                - reps: int (number of repetitions)
                - avg_form_score: float (0-10 scale)
                - duration: float (seconds)
        """
        # Validate required fields
        required_fields = ['timestamp', 'exercise_type', 'reps', 'avg_form_score', 'duration']
        if not all(field in session_data for field in required_fields):
            return
        
        # Add to sessions list
        self.sessions.append(session_data)
        
        # Update DataFrame
        new_row = pd.DataFrame([session_data])
        self.session_history = pd.concat([self.session_history, new_row], ignore_index=True)
    
    def get_progress_metrics(self, exercise_type: Optional[str] = None) -> Optional[Dict]:
        """
        Get comprehensive progress metrics for a specific exercise type.
        
        Args:
            exercise_type: Filter by exercise type, or None for all exercises
            
        Returns:
            Dictionary containing various metrics, or None if no data
        """
        if len(self.session_history) == 0:
            return None
        
        # Filter by exercise type if specified
        if exercise_type:
            df = self.session_history[self.session_history['exercise_type'] == exercise_type]
        else:
            df = self.session_history
        
        if len(df) == 0:
            return None
        
        # Calculate metrics
        total_reps = int(df['reps'].sum())
        total_sessions = len(df)
        avg_form_score = float(df['avg_form_score'].mean())
        avg_duration = float(df['duration'].mean())
        
        # Find best session
        best_session_idx = df['avg_form_score'].idxmax()
        best_session = df.loc[best_session_idx]
        
        # Calculate improvement areas
        improvement_areas = self._identify_improvement_areas(df)
        
        # Weekly progress
        weekly_stats = self._get_weekly_stats(df)
        
        return {
            'total_reps': total_reps,
            'total_sessions': total_sessions,
            'avg_form_score': avg_form_score,
            'avg_duration': avg_duration,
            'best_session': {
                'date': best_session['timestamp'].strftime('%Y-%m-%d %H:%M'),
                'reps': int(best_session['reps']),
                'form_score': float(best_session['avg_form_score']),
                'duration': float(best_session['duration'])
            },
            'improvement_areas': improvement_areas,
            'weekly_stats': weekly_stats
        }
    
    def _identify_improvement_areas(self, df: pd.DataFrame) -> List[str]:
        """
        Identify areas where the user could improve.
        
        Args:
            df: DataFrame with session data
            
        Returns:
            List of improvement suggestions
        """
        areas = []
        
        # Check form score consistency
        if len(df) >= 3:
            recent_scores = df['avg_form_score'].tail(3).tolist()
            if statistics.stdev(recent_scores) > 2.0:
                areas.append("Form consistency - scores vary significantly")
        
        # Check if form score is below optimal
        avg_form = df['avg_form_score'].mean()
        if avg_form < 7.0:
            areas.append("Form quality - aim for higher accuracy")
        
        # Check session frequency
        if len(df) >= 2:
            df_sorted = df.sort_values('timestamp')
            time_gaps = df_sorted['timestamp'].diff().dt.total_seconds() / 3600  # hours
            avg_gap = time_gaps.mean()
            
            if avg_gap > 48:  # More than 2 days between sessions
                areas.append("Session frequency - try to exercise more regularly")
        
        # Check duration consistency
        if len(df) >= 3:
            recent_durations = df['duration'].tail(3).tolist()
            if any(d < 30 for d in recent_durations):
                areas.append("Session duration - longer sessions may improve results")
        
        return areas
    
    def _get_weekly_stats(self, df: pd.DataFrame) -> Dict:
        """
        Calculate statistics for the past week.
        
        Args:
            df: DataFrame with session data
            
        Returns:
            Dictionary with weekly statistics
        """
        if len(df) == 0:
            return {
                'sessions_this_week': 0,
                'reps_this_week': 0,
                'avg_form_this_week': 0.0
            }
        
        # Get sessions from last 7 days
        week_ago = datetime.now() - timedelta(days=7)
        weekly_df = df[df['timestamp'] >= week_ago]
        
        return {
            'sessions_this_week': len(weekly_df),
            'reps_this_week': int(weekly_df['reps'].sum()) if len(weekly_df) > 0 else 0,
            'avg_form_this_week': float(weekly_df['avg_form_score'].mean()) if len(weekly_df) > 0 else 0.0
        }
    
    def get_exercise_summary(self) -> Dict:
        """
        Get an overall summary of all exercises.
        
        Returns:
            Dictionary with summary statistics
        """
        if len(self.session_history) == 0:
            return {
                'total_sessions': 0,
                'total_reps': 0,
                'exercises_types': [],
                'total_time_spent': 0.0
            }
        
        return {
            'total_sessions': len(self.session_history),
            'total_reps': int(self.session_history['reps'].sum()),
            'exercise_types': self.session_history['exercise_type'].unique().tolist(),
            'total_time_spent': float(self.session_history['duration'].sum()),
            'avg_form_score': float(self.session_history['avg_form_score'].mean())
        }
    
    def export_data(self, filepath: str):
        """
        Export session history to CSV file.
        
        Args:
            filepath: Path where to save the CSV file
        """
        if len(self.session_history) > 0:
            self.session_history.to_csv(filepath, index=False)
            return True
        return False
    
    def import_data(self, filepath: str):
        """
        Import session history from CSV file.
        
        Args:
            filepath: Path to the CSV file
        """
        try:
            imported_df = pd.read_csv(filepath)
            imported_df['timestamp'] = pd.to_datetime(imported_df['timestamp'])
            self.session_history = pd.concat([self.session_history, imported_df], ignore_index=True)
            self.sessions = self.session_history.to_dict('records')
            return True
        except Exception as e:
            print(f"Error importing data: {e}")
            return False
    
    def clear_history(self):
        """Clear all session history."""
        self.sessions = []
        self.session_history = pd.DataFrame(columns=[
            'timestamp', 'exercise_type', 'reps', 'avg_form_score', 'duration'
        ])