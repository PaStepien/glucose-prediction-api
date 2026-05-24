import pandas as pd

def build_assistant_context(df):
    
    if df is None or df.empty:
        return {
            "time_since_entry": "unknown",
            "trend": "unknown",
            "recent_change": "unknown",
            "carbs": "unknown",
            "meal_time": "unknown",
            "insulin": "unknown",
            "insulin_time": "unknown",
            "steps": "unknown",
        }
    
    last = df.iloc[-1]
    prev = df.iloc[-6] if len(df) >= 6 else last

    diff = float(last["glucose"] - prev["glucose"])
    if diff > 15:
        trend = "rapidly rising"
    elif diff > 5:
        trend = "rising"
    elif diff < -15:
        trend = "rapidly falling"
    elif diff < -5:
        trend = "falling"
    else:
        trend = "stable"

    recent_change = f"{diff:+.1f} mg/dL in last 30 minutes"

    carbs_rows = df[df["carbs"] > 0]
    bolus_rows = df[df["bolus_raw"] > 0]

    meal_time = (
        int((df.index[-1] - carbs_rows.index[-1]).total_seconds() / 60)
        if not carbs_rows.empty else "unknown"
    )
    insulin_time = (
        int((df.index[-1] - bolus_rows.index[-1]).total_seconds() / 60)
        if not bolus_rows.empty else "unknown"
    )
        
    carbs = float(carbs_rows["carbs"].iloc[-1]) if not carbs_rows.empty else "unknown"
    insulin = float(bolus_rows["bolus_raw"].iloc[-1]) if not bolus_rows.empty else "unknown"

    steps_value = float(last["steps"])
    if steps_value < 50:
        steps = "low"
    elif steps_value < 200:
        steps = "moderate"
    else:
        steps = "high"
        
    time_since_entry = int((pd.Timestamp("2026-05-23 13:30:00+00:00") - df.index[-1]).total_seconds() / 60)

    return {
        "trend": trend,
        "recent_change": recent_change,
        "carbs": carbs,
        "meal_time": meal_time,
        "insulin": insulin,
        "insulin_time": insulin_time,
        "steps": steps,
        "time_since_entry": time_since_entry,
        "glucose_sequence": df["glucose"].tolist()[-36:],
    }