#!/usr/bin/env python3
"""
Advanced Health Risk Monitoring System
--------------------------------------
Features:
- Realistic vital signs simulation (glucose, HR, BP)
- Personalized risk scoring
- Trend detection
- SQLite storage
- Live matplotlib dashboard
- ML-based glucose prediction
- Email alerts (optional)
"""

import random
import time
import sqlite3
import threading
from collections import deque
from datetime import datetime, timedelta
import numpy as np

# === Optional imports with graceful fallback ===
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not installed – live plotting disabled.")

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not installed – using simple trend prediction.")

# ================================================
# 1. REALISTIC DATA SIMULATION (time‑of‑day patterns)
# ================================================
class HealthDataSimulator:
    """Simulates glucose, heart rate, and blood pressure with realistic daily patterns."""
    
    def __init__(self, patient):
        self.patient = patient
        self.last_glucose = 100
        self.last_hr = 70
        
    def get_current_hour(self):
        return datetime.now().hour
    
    def simulate_glucose(self):
        hour = self.get_current_hour()
        # Base glucose depends on diabetes status
        base = 120 if self.patient.has_diabetes else 90
        
        # Meal effects
        if 7 <= hour <= 9:      # breakfast
            base += random.uniform(20, 45)
        elif 12 <= hour <= 14:  # lunch
            base += random.uniform(30, 55)
        elif 18 <= hour <= 20:  # dinner
            base += random.uniform(20, 40)
        
        # Random variation + trend towards previous value (autocorrelation)
        noise = random.gauss(0, 5)
        new_glucose = 0.7 * self.last_glucose + 0.3 * base + noise
        self.last_glucose = max(40, min(400, new_glucose))   # clamp to physiological range
        return int(self.last_glucose)
    
    def simulate_heart_rate(self):
        hour = self.get_current_hour()
        # Higher during day, lower at night
        if 0 <= hour < 6:
            base = 55
        elif 6 <= hour < 22:
            base = 75
        else:
            base = 65
        
        # Activity burst (random)
        if random.random() < 0.2:
            base += random.randint(10, 30)
        
        noise = random.gauss(0, 3)
        new_hr = 0.8 * self.last_hr + 0.2 * base + noise
        self.last_hr = max(40, min(150, new_hr))
        return int(self.last_hr)
    
    def simulate_blood_pressure(self):
        # Simple systolic BP simulation
        hour = self.get_current_hour()
        base_sys = 110
        if hour < 8:
            base_sys = 105
        elif hour > 20:
            base_sys = 115
        noise = random.gauss(0, 8)
        sys = base_sys + noise
        dia = sys * 0.6 + random.gauss(0, 5)
        return int(sys), int(dia)
    
    def get_all_vitals(self):
        glucose = self.simulate_glucose()
        heart_rate = self.simulate_heart_rate()
        bp_sys, bp_dia = self.simulate_blood_pressure()
        return {
            'glucose': glucose,
            'heart_rate': heart_rate,
            'bp_systolic': bp_sys,
            'bp_diastolic': bp_dia,
            'timestamp': datetime.now()
        }

# ================================================
# 2. PERSONALIZED THRESHOLDS & RISK SCORING
# ================================================
class Patient:
    def __init__(self, age, has_diabetes=False, name="Anonymous"):
        self.name = name
        self.age = age
        self.has_diabetes = has_diabetes
        
        # Personalized thresholds
        self.glucose_target = 130 if has_diabetes else 100
        self.glucose_high_alert = 180 if has_diabetes else 140
        self.max_hr = 220 - age
        self.min_hr = 50
        self.bp_sys_alert = 130
        self.bp_dia_alert = 85

def compute_risk_score(vitals, patient):
    """Returns a risk score (0-100) and a list of active alerts."""
    score = 0
    alerts = []
    
    # Glucose
    g = vitals['glucose']
    if g > patient.glucose_high_alert:
        alerts.append(f"High glucose: {g} mg/dL (threshold {patient.glucose_high_alert})")
        score += 40
    elif g > patient.glucose_target:
        alerts.append(f"Elevated glucose: {g} mg/dL")
        score += 20
    
    # Heart rate
    hr = vitals['heart_rate']
    if hr > patient.max_hr:
        alerts.append(f"Tachycardia: HR {hr} bpm (max {patient.max_hr})")
        score += 30
    elif hr < patient.min_hr:
        alerts.append(f"Bradycardia: HR {hr} bpm")
        score += 30
    
    # Blood pressure
    sys = vitals['bp_systolic']
    dia = vitals['bp_diastolic']
    if sys > patient.bp_sys_alert:
        alerts.append(f"High systolic BP: {sys} mmHg")
        score += 20
    if dia > patient.bp_dia_alert:
        alerts.append(f"High diastolic BP: {dia} mmHg")
        score += 20
    
    return min(score, 100), alerts

# ================================================
# 3. TREND DETECTION (using sliding window)
# ================================================
class TrendDetector:
    def __init__(self, window_size=6, threshold_rise=20):
        self.glucose_history = deque(maxlen=window_size)
        self.threshold = threshold_rise  # mg/dL rise over window
    
    def add_reading(self, glucose):
        self.glucose_history.append(glucose)
    
    def rapid_rise_detected(self):
        if len(self.glucose_history) < 3:
            return False
        # Check last 3 vs 3 before that (if enough data)
        recent = list(self.glucose_history)[-3:]
        older = list(self.glucose_history)[:-3] if len(self.glucose_history) >= 6 else None
        if older:
            if (recent[-1] - older[0]) > self.threshold:
                return True
        # Alternatively, check consecutive increases
        increases = all(recent[i] < recent[i+1] for i in range(len(recent)-1))
        if increases and (recent[-1] - recent[0]) > 15:
            return True
        return False

# ================================================
# 4. SQLITE DATABASE FOR PERSISTENT STORAGE
# ================================================
class Database:
    def __init__(self, db_path="health_monitor.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
    
    def create_tables(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS vitals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                glucose INTEGER,
                heart_rate INTEGER,
                bp_systolic INTEGER,
                bp_diastolic INTEGER,
                risk_score INTEGER
            )
        ''')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                alert_message TEXT,
                risk_score INTEGER
            )
        ''')
        self.conn.commit()
    
    def insert_vitals(self, vitals, risk_score):
        self.conn.execute(
            "INSERT INTO vitals (timestamp, glucose, heart_rate, bp_systolic, bp_diastolic, risk_score) VALUES (?, ?, ?, ?, ?, ?)",
            (vitals['timestamp'].isoformat(), vitals['glucose'], vitals['heart_rate'],
             vitals['bp_systolic'], vitals['bp_diastolic'], risk_score)
        )
        self.conn.commit()
    
    def insert_alert(self, message, risk_score):
        self.conn.execute(
            "INSERT INTO alerts (timestamp, alert_message, risk_score) VALUES (?, ?, ?)",
            (datetime.now().isoformat(), message, risk_score)
        )
        self.conn.commit()
    
    def close(self):
        self.conn.close()

# ================================================
# 5. ALERT SYSTEM (print + optional email)
# ================================================
class AlertSystem:
    def __init__(self, email_config=None):
        self.email_config = email_config  # {'from':..., 'to':..., 'password':...}
    
    def send_alert(self, message, risk_score):
        """Print alert and optionally send email."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] RISK {risk_score}: {message}"
        print("\n🔔 ALERT:", full_msg)
        
        if self.email_config:
            try:
                import smtplib
                from email.mime.text import MIMEText
                msg = MIMEText(full_msg)
                msg['Subject'] = f"Health Alert - Risk {risk_score}"
                msg['From'] = self.email_config['from']
                msg['To'] = self.email_config['to']
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                    server.login(self.email_config['from'], self.email_config['password'])
                    server.send_message(msg)
                print("   Email alert sent.")
            except Exception as e:
                print(f"   Email failed: {e}")

# ================================================
# 6. MACHINE LEARNING PREDICTION (glucose forecast)
# ================================================
class GlucosePredictor:
    def __init__(self, lookback=5):
        self.lookback = lookback
        self.history = deque(maxlen=50)  # store (timestamp, glucose)
        self.model = None
        if SKLEARN_AVAILABLE:
            self.model = LinearRegression()
    
    def add_reading(self, glucose):
        self.history.append((datetime.now(), glucose))
    
    def predict_next_glucose(self):
        """Predict glucose 30 minutes ahead using linear regression on last 20 points."""
        if len(self.history) < self.lookback + 5:
            return None
        
        # Prepare features: time difference in minutes and previous glucose values
        data = list(self.history)
        X, y = [], []
        for i in range(len(data) - self.lookback):
            # Feature: last 'lookback' glucose readings + minute of day
            features = [data[j][1] for j in range(i, i+self.lookback)]
            features.append(data[i+self.lookback-1][0].minute)  # minute of day
            X.append(features)
            y.append(data[i+self.lookback][1])
        
        if len(X) < 10:
            return None
        
        if SKLEARN_AVAILABLE and self.model:
            self.model.fit(X, y)
            # Latest features
            last_features = [data[-j][1] for j in range(1, self.lookback+1)][::-1]
            last_features.append(datetime.now().minute)
            pred = self.model.predict([last_features])[0]
            return int(pred)
        else:
            # Simple trend: average of last 3 differences
            diffs = [data[-i][1] - data[-i-1][1] for i in range(1, 4)]
            trend = np.mean(diffs)
            return int(data[-1][1] + trend)

# ================================================
# 7. LIVE PLOTTING (using matplotlib in a separate thread)
# ================================================
class LivePlotter:
    def __init__(self, max_points=100):
        if not MATPLOTLIB_AVAILABLE:
            self.enabled = False
            return
        self.enabled = True
        self.max_points = max_points
        self.glucose_data = deque(maxlen=max_points)
        self.hr_data = deque(maxlen=max_points)
        self.time_data = deque(maxlen=max_points)
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.ax1.set_ylabel('Glucose (mg/dL)')
        self.ax2.set_ylabel('Heart Rate (bpm)')
        self.ax2.set_xlabel('Time (seconds)')
        self.line1, = self.ax1.plot([], [], 'r-', label='Glucose')
        self.line2, = self.ax2.plot([], [], 'b-', label='Heart Rate')
        self.ax1.legend()
        self.ax2.legend()
        self.ax1.grid(True)
        self.ax2.grid(True)
    
    def update(self, glucose, hr):
        if not self.enabled:
            return
        self.glucose_data.append(glucose)
        self.hr_data.append(hr)
        t = len(self.glucose_data)
        self.time_data.append(t)
        self.line1.set_data(list(self.time_data), list(self.glucose_data))
        self.line2.set_data(list(self.time_data), list(self.hr_data))
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        plt.pause(0.01)
    
    def close(self):
        if self.enabled:
            plt.ioff()
            plt.close()

# ================================================
# 8. MAIN MONITORING LOOP (integrates everything)
# ================================================
class HealthMonitor:
    def __init__(self, patient, email_config=None):
        self.patient = patient
        self.simulator = HealthDataSimulator(patient)
        self.db = Database()
        self.alert_system = AlertSystem(email_config)
        self.trend_detector = TrendDetector()
        self.predictor = GlucosePredictor()
        self.plotter = LivePlotter()
        self.running = True
    
    def run(self, interval_seconds=5):
        """Main loop – collects data, analyses, stores, and plots."""
        print(f"Starting health monitor for {self.patient.name} (age {self.patient.age}, diabetes: {self.patient.has_diabetes})")
        print("Press Ctrl+C to stop.\n")
        
        try:
            while self.running:
                # 1. Get simulated vitals
                vitals = self.simulator.get_all_vitals()
                glucose = vitals['glucose']
                hr = vitals['heart_rate']
                
                # 2. Trend detection
                self.trend_detector.add_reading(glucose)
                if self.trend_detector.rapid_rise_detected():
                    self.alert_system.send_alert(f"Rapid glucose rise detected! Last readings: {list(self.trend_detector.glucose_history)}", risk_score=60)
                
                # 3. Risk scoring
                risk_score, alerts = compute_risk_score(vitals, self.patient)
                
                # 4. ML prediction (show every 5 cycles)
                self.predictor.add_reading(glucose)
                pred = self.predictor.predict_next_glucose()
                pred_msg = f" | Predicted next glucose: {pred}" if pred else ""
                
                # 5. Print status
                status = f"Glucose: {glucose} mg/dL | HR: {hr} bpm | BP: {vitals['bp_systolic']}/{vitals['bp_diastolic']} | Risk: {risk_score}{pred_msg}"
                print(status)
                
                # 6. Raise alerts based on risk score and individual alerts
                for alert in alerts:
                    self.alert_system.send_alert(alert, risk_score)
                    self.db.insert_alert(alert, risk_score)
                
                if risk_score >= 70:
                    self.alert_system.send_alert("High overall risk score! Consider medical attention.", risk_score)
                    self.db.insert_alert("High overall risk", risk_score)
                
                # 7. Store in database
                self.db.insert_vitals(vitals, risk_score)
                
                # 8. Update live plot
                self.plotter.update(glucose, hr)
                
                # 9. Wait for next reading
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
        finally:
            self.shutdown()
    
    def shutdown(self):
        self.running = False
        self.db.close()
        self.plotter.close()
        print("Resources released. Goodbye.")

# ================================================
# 9. ENTRY POINT – CONFIGURE PATIENT & START
# ================================================
if __name__ == "__main__":
    # Create a patient (age, diabetes status, name)
    patient = Patient(age=65, has_diabetes=True, name="John Doe")
    
    # Optional email alerts – set your credentials or leave as None
    email_config = None
    # Uncomment and fill to enable email:
    # email_config = {
    #     'from': 'your_email@gmail.com',
    #     'to': 'doctor@example.com',
    #     'password': 'your_app_password'
    # }
    
    monitor = HealthMonitor(patient, email_config)
    monitor.run(interval_seconds=5)   # check every 5 seconds