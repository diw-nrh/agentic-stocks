from datetime import datetime, timedelta
import yfinance as yf

end = datetime.now()
start = end - timedelta(days=1)
def get_stocks(symbols:str,start:str,end:str,period:str):
    stock = yf.Ticker(symbols)
    if period == "scheduled":
        data = stock.history(end=end)
        data.tail(1)
    elif period == "range":
        data = stock.history(start=start, end=end)
    else:
        data = stock.history(period="1d")
    try:   
        return data.reset_index().to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}