from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db import get_db
from app.routes.dashboard import router as dashboard_router
from app.routes.auth import router as auth_router
from app.routes.ml import router as ml_router

app = FastAPI(title="Admin Dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard_router)
app.include_router(auth_router)
app.include_router(ml_router)


@app.get("/")
def root():
    return {"message": "Backend is running"}


@app.get("/test-users")
def test_users(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT COUNT(*) FROM users")).fetchone()
    return {"total_users": result[0]}
