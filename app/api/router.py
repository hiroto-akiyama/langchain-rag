"""APIルーター集約"""
from fastapi import APIRouter

from app.api import chat, documents, search

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(documents.router)
api_router.include_router(search.router)
api_router.include_router(chat.router)
