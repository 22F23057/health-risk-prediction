import random
import time

# Simulated data for blood sugar and heart rate
def get_health_data():
    # Simulate random health data (blood sugar and heart rate)
    blood_sugar_level = random.randint(70, 200)  # Random blood sugar level
    heart_rate = random.randint(50, 120)         # Random heart rate (normal range)
    
    return blood_sugar_level, heart_rate

def health_risk_prediction():
    # Define threshold values
    diabetes_threshold = 140  # Changed threshold for diabetes risk
    low_heart_rate = 60      # Below this heart rate is considered too low
    high_heart_rate = 100    # Above this heart rate is considered too high

    while True:  # Simulating continuous real-time monitoring
        # Step 1: Get simulated health data
        blood_sugar_level, heart_rate = get_health_data()

        # Step 2: Check for health risks
        if blood_sugar_level > diabetes_threshold:
            print(f"Blood Sugar Level: {blood_sugar_level} - Risk of Diabetes")
        elif heart_rate < low_heart_rate or heart_rate > high_heart_rate:
            print(f"Heart Rate: {heart_rate} - Risk of Heart Disease")
        else:
            print(f"Blood Sugar Level: {blood_sugar_level}, Heart Rate: {heart_rate} - Health is Normal")
        
        # Simulating real-time data collection with a small delay
        time.sleep(5)  # Wait for 5 seconds before checking again

# Run the health risk prediction system
health_risk_prediction()
