---
name: news_skill
description: Get news data for countries and cities, including current conditions, forecasts, and historical trends.
allowed-tools: [news]
---

# News Skill

## Instructions
1. Use this tool whenever you need to check news data.
2. **News**: Provide the news topic (e.g., "weather", "stock").
3. **Location**: Provide the location (e.g., "thailand", "usa").
4. **Start**: Provide the start date in YYYY-MM-DD format.
5. **End**: Provide the end date in YYYY-MM-DD format.
6. **Period**: Specify the time frame you are looking for:
   - `current`: Current news data.
   - `scheduled`: Scheduled news data.
   - `range`: Range news data.
7. **Day**: Specify the number of days for historical data.

## Examples
- "What is the news range data for weather in thailand between March 20 and March 24?" -> `news(news="weather", location="thailand", start="2024-03-20", end="2024-03-24", period="range")`
- "What is the scheduled news data for weather in thailand up to March 24?" -> `news(news="weather", location="thailand", end="2024-03-24", period="scheduled")`
- "What is the current news data for weather in thailand?" -> `news(news="weather", location="thailand", period="current")`
- "What is the news range data for weather in america between March 20 and March 24?" -> `news(news="weather", location="america", start="2024-03-20", end="2024-03-24", period="range")`
- "What is the scheduled news data for weather in america up to March 24?" -> `news(news="weather", location="america", end="2024-03-24", period="scheduled")`
- "What is the current news data for weather in america?" -> `news(news="weather", location="america", period="current")`
- "What is the scheduled news data for weather in songkhla?" -> `news(news="weather", location="songkhla", period="scheduled")`
- "What is the news range data for stock in songkhla between March 20 and March 24?" -> `news(news="stock", location="songkhla", start="2024-03-20", end="2024-03-24", period="range")`
- "What is the scheduled news data for stock in america?" -> `news(news="stock", location="america", period="scheduled")`
- "What is the current news data for stock in america?" -> `news(news="stock", location="america", period="current")`