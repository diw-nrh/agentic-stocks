from fastapi import FastAPI
from memory.database import create_db_and_tables
from memory.skills.router import router as skills_router

app = FastAPI(title="Agentic Memory System", version="1.0.0")

@app.on_event("startup")
def on_startup():
    try:
        create_db_and_tables()
        print("Memory DB tables created successfully.")
    except Exception as e:
        print(f"Warning: DB not ready yet: {e}")

app.include_router(skills_router)

@app.get("/")
def root():
    return {"status": "Memory Service is running"}
