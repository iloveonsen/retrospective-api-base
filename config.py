from pydantic import Field
from pydantic_settings import BaseSettings

import os
from dotenv import load_dotenv

load_dotenv()


DB_USERNAME = os.environ.get("DB_USERNAME") 
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_URL = f"postgresql+asyncpg://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
SUMMARY_MODEL_PATH = os.environ.get("SUMMARY_MODEL_PATH")
RETROSPECTIVE_MODEL_PATH = os.environ.get("RETROSPECTIVE_MODEL_PATH")


class Config(BaseSettings):
    db_url: str = Field(default=DB_URL, env="DB_URL")
    summary_model_path: str = Field(default=SUMMARY_MODEL_PATH, env="SUMMARY_MODEL_PATH") # transformers
    retrospective_model_path: str = Field(default=RETROSPECTIVE_MODEL_PATH, env="RETROSPECTIVE_MODEL_PATH") # langchain llamacpp gguf
    


config = Config()
