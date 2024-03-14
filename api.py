from fastapi import APIRouter, Depends

from sqlalchemy.ext.asyncio import AsyncSession
from langchain.chains import LLMChain

import numpy as np

from database import Retrospective, get_db_session
from model import get_tf_model, get_llm_chain
from schemas import (
    ModelGenerationRequest,
    ModelGenerationResponse,
    RetrospectiveRequest, 
    RetrospectiveResponse
)


router = APIRouter()



@router.post("/generate/tf")
async def generate_tf_model(request: ModelGenerationRequest,
                            tf_model: tuple = Depends(get_tf_model)) -> ModelGenerationResponse:
    device, tokenizer, model = tf_model
    inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask, 
        max_length=64, num_beams=5, length_penalty=1.2, use_cache=True, early_stopping=True
    ).detach().cpu()
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return ModelGenerationResponse(text=text)



@router.post("/generate/chain")
async def generate_chain(request: ModelGenerationRequest,
                         llm_chain: LLMChain = Depends(get_llm_chain)) -> ModelGenerationResponse:
    
    context = {
        "input": "너는 사용자에게 도움을 주는 AI야",
        "instruction": request.text
    }
    
    result = await llm_chain.ainvoke(input=context)

    return ModelGenerationResponse(text=result.get("text"))



@router.post("/predict")
async def predict(request: RetrospectiveRequest,
                  tf_model:tuple = Depends(get_tf_model),
                  llm_chain: LLMChain = Depends(get_llm_chain),
                  db: AsyncSession = Depends(get_db_session)) -> RetrospectiveResponse:
    device, tokenizer, model = tf_model

    user_inputs = request.user_inputs
    user_inputs_length = len(user_inputs)

    chunks = []
    turns_per_chunk = 5
    total_chunks = user_inputs_length // turns_per_chunk if user_inputs_length % turns_per_chunk == 0 else user_inputs_length // turns_per_chunk + 1
    for i in range(total_chunks):
        start = i * turns_per_chunk
        end = (i + 1) * turns_per_chunk if (i + 1) * turns_per_chunk < user_inputs_length else user_inputs_length
        chunks.append("[BOS]" + "[SEP]".join(user_inputs[start:end]) + "[EOS]")

    
    inputs = tokenizer(chunks, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask, 
        max_length=64, num_beams=5, length_penalty=1.2, use_cache=True, early_stopping=True).detach().cpu()
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = "".join(summary)

    
    context = {
        "username": request.username,
        "summary": summary,
    }

    result = await llm_chain.ainvoke(input=context)
    result = result.get("text").strip()

    retrospective = Retrospective(
        username=request.username,
        text=result
    )
    db.add(retrospective)
    await db.commit()

    return RetrospectiveResponse(text=result)
    
    



