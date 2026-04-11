from fastapi import FastAPI, Query
from services.news_service.service import get_news

app = FastAPI()
@app.get("/news")
def news_endpoint(
    news: str = Query(..., description="Query topic for news"),
    location: str = Query(..., description="Location for news"),
    start: str = Query(None, description="Date in YYYY-MM-DD format for historical data"),
    end: str = Query(None, description="Date in YYYY-MM-DD format for historical data"),
    period: str = Query("current", enum=["current", "scheduled", "range"]),
    day: str = "30",
):
    return get_news(news,location,start,end,period,day)