---
name: weather
description: use this skill when you want to find the weather information of a location
allowed-tools: [weather]
---

# weather

## Overview
This skill allows the agent to find the weather information of a location.

## Instructions
1. Use the `weather` tool to find the weather information of a location.
2. The `weather` tool requires the `location` parameter.
3. The `weather` tool returns the weather information of the specified location.

## Constraints
1. The `location` parameter is required.
2. The `weather` tool can only find the weather information of one location at a time.
3. The `weather` tool can only find the weather information of the current date.

## Examples
- User: "What is the weather in Bangkok?" -> Call `weather(location="Bangkok")`
- User: "Do I need to bring an umbrella to Phuket today?" -> Call `weather(location="Phuket")`

## Response Guidelines
1. If the temperature is higher than 30°C, recommend the user to drink plenty of water.
2. If the description contains the word "rain", remind the user to bring an umbrella.