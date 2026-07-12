from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import corroborate, dataviz, predict

app = FastAPI(title="Fake News Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dataviz.router)
app.include_router(predict.router)
app.include_router(corroborate.router)


@app.get("/health")
def health():
    return {"status": "ok"}
