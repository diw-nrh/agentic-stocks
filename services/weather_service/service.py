import requests
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.config import Config

def get_weather(location: str, period: str = "current"):
    api_key = Config.Weather.OPENWEATHER_API_KEY
    base_url = "http://api.weatherapi.com/v1"
    
    if period == "forecast":
        url = f"{base_url}/forecast.json?key={api_key}&q={location}&days=7&lang=th"
    elif period == "historical":
        url = f"{base_url}/history.json?key={api_key}&q={location}&dt=2024-03-24&lang=th"
    else:
        url = f"{base_url}/current.json?key={api_key}&q={location}&lang=th"

    try:
        response = requests.get(url)
        data = response.json()
        
        if "error" in data:
            return {"error": data["error"]["message"]}
            
        return {
            "display_name": f"{data.get('location', {}).get('name', 'Unknown')}, {data.get('location', {}).get('country', 'Unknown')}",
            "temp": data.get("current", {}).get("temp_c") if period != "forecast" else "See forecast data",
            "condition": data.get("current", {}).get("condition", {}).get("text") if period != "forecast" else "See forecast data",
            "full_data": data
        }
    except Exception as e:
        return {"error": str(e)}