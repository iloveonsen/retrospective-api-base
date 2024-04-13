from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from sqlmodel import SQLModel
from loguru import logger

from api import router
from database import engine, Retrospective
from model import load_tf_model, load_llm_chain
from config import config

import os



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application")

    logger.info(f"DB_USERNAME: {os.environ.get('DB_USERNAME')}")
    logger.info(f"DB_PASSWORD: {os.environ.get('DB_PASSWORD')}")
    logger.info(f"DB_HOST: {os.environ.get('DB_HOST')}")
    logger.info(f"DB_NAME: {os.environ.get('DB_NAME')}")

    logger.info(f"Creating database connection to {config.db_url}")
    async with engine.begin() as conn:
        try:
            await conn.run_sync(SQLModel.metadata.create_all)
            logger.info("Table created")
        except Exception as e:
            logger.error(f"Error creaing table: {e}")

    logger.info("Loading transformers model")
    load_tf_model()
    
    logger.info("Loading llm chain")
    load_llm_chain()

    yield

    logger.info("Shutting down")
    await engine.dispose()



app = FastAPI(lifespan=lifespan)
app.include_router(router=router)


@app.get("/")
async def root():
    return JSONResponse(content={"message": "This is the root of retrospective service api."}, status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8003, reload=True)