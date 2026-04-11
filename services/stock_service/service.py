import yfinance as yf


def get_stocks(symbols: str, start: str, end: str, period: str):
    try:
        stock = yf.Ticker(symbols)

        if period == "scheduled":
            history_df = stock.history(end=end).tail(1)
        elif period == "range":
            history_df = stock.history(start=start, end=end)
        else:
            history_df = stock.history(period="1d")

        records = history_df.reset_index().to_dict(orient="records")
        for row in records:
            row["symbol"] = symbols

        return records
    except Exception as e:
        return {"error": str(e)}