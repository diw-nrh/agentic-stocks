---
name: stock-skill
description: Get stock data for countries and cities, including current conditions, forecasts, and historical trends.
allowed-tools: [stock]
---

# Stock Skill

## Instructions
1. Use this tool whenever you need to check stock data.
2. **Symbols**: Provide the stock symbols (e.g., "AAPL", "GOOGL").
3. **Start**: Provide the start date in YYYY-MM-DD format.
4. **End**: Provide the end date in YYYY-MM-DD format.
5. **Period**: Specify the time frame you are looking for:
   - `current`: Current stock data.
   - `scheduled`: Scheduled stock data.
   - `range`: Range stock data.

## Examples
- "What is the stock range data for AAPL between March 20 and March 24?" -> `stock(symbols="AAPL", start="2024-03-20", end="2024-03-24", period="range")`
- "What is the scheduled stock data for AAPL up to March 24?" -> `stock(symbols="AAPL", end="2024-03-24", period="scheduled")`
- "What is the current stock data for AAPL?" -> `stock(symbols="AAPL", period="current")`

## rules
- Symbols can only be accessed one at a time; they cannot be accessed simultaneously.