from fastapi import APIRouter, Depends

from sqlalchemy.ext.asyncio import AsyncSession
from langchain.chains import LLMChain

from database import Retrospective, get_db_session
from model import get_chain
from schemas import (
    ModelGenerationRequest,
    ModelGenerationResponse,
    RetrospectiveRequest, 
    RetrospectiveResponse
)


router = APIRouter()




@router.post("/generate/chain")
async def generate_chain(request: ModelGenerationRequest,
                         chain: LLMChain = Depends(get_chain)) -> ModelGenerationResponse:
    
    context = {
        "input": "너는 사용자에게 도움을 주는 AI야",
        "instruction": request.text
    }
    
    result = await chain.ainvoke(input=context, max_tokens=100, temperature=0.6, top_p=1, verbose=True)

    return ModelGenerationResponse(text=result.get("text"))


@router.post("/predict")
async def predict(request: RetrospectiveRequest,
                  chain: LLMChain = Depends(get_chain),
                  db: AsyncSession = Depends(get_db_session)) -> RetrospectiveResponse:
    

    return RetrospectiveResponse(text="response")
    
    



