from pydantic import BaseModel, Field
from typing import List


class ModelGenerationRequest(BaseModel):
    text: str

class ModelGenerationResponse(BaseModel):
    text: str


class RetrospectiveRequest(BaseModel):
    username: str
    user_inputs: List[str]
    bot_responses: List[str]


class RetrospectiveResponse(BaseModel):
    text: str