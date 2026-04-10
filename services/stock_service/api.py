from fastapi import FastAPI, Query
from services.stock_service.service import get_stocks

app = FastAPI()

@app.get("/stocks")
def stocks_endpoint(
    symbols: str,
    start: str = Query(None, description="Date in YYYY-MM-DD format for historical data"),
    end: str = Query(None, description="Date in YYYY-MM-DD format for historical data"),
    period: str = Query("current", enum=["current", "scheduled", "range"]),
):
    return get_stocks(symbols,start,end,period)