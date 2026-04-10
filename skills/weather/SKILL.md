---
name: weather_skill
description: Get weather data for countries and cities, including current conditions, forecasts, and historical trends.
allowed-tools: [weather]
---

# Weather Skill

## Instructions
1. Use this tool whenever you need to check weather information.
2. **Location**: Provide the country name (e.g., Thailand) or city/province name (e.g., Phuket).
3. **Period**: Specify the exact time frame you are looking for:
   - `current`: Current weather conditions right now.
   - `forecast`: 7-day weather forecast (use for future trends).
   - `historical`: Historical weather data (use for past statistics). You CAN and SHOULD also specify `dt` in YYYY-MM-DD format (e.g., "2024-03-24"). By default it gives yesterday's weather.

## Examples
- "What is the weather like in Thailand right now?" -> `weather(location="Thailand", period="current")`
- "Show me the weather forecast for Chiang Mai next week." -> `weather(location="Chiang Mai", period="forecast")`
- "What was the weather in Phuket on March 15, 2024?" -> `weather(location="Phuket", period="historical", dt="2024-03-15")`
- - "What is the weather like in Thailand right now?" -> `weather(location="Thailand", period="current")`