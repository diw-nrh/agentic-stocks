import feedparser
from datetime import datetime

def get_news(news:str, location:str, start:str, end:str, period:str, day:str):
    url = f"https://news.google.com/rss/search?q={news}+{location}+when:{day}d&hl=en&gl=US&ceid=US:en"
    feed = feedparser.parse(url)

    start_date = datetime.strptime(start, "%Y-%m-%d") if start else None
    end_date = datetime.strptime(end, "%Y-%m-%d") if end else None

    results = []

    for entry in feed.entries:
        published = datetime(*entry.published_parsed[:6])

        if period == "range":
            if start_date and end_date and start_date <= published <= end_date:
                results.append(entry)

        elif period == "scheduled":
            target = end_date or start_date
            if target and published.date() == target.date():
                results.append(entry)

        else:
            return [{
                "title": entry.title,
                "published": published.isoformat(),
            }]

    return results