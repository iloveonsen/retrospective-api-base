from fastapi import APIRouter, WebSocket, Depends, WebSocketDisconnect

from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain.chains import LLMChain

from loguru import logger
import numpy as np
from typing import Optional, List

from database import Retrospective, get_db_session
from model import get_tf_model, get_llm_chain, make_summary, async_func
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
        "username": request.username,
        "summary": request.text
    }
    
    result = await llm_chain.ainvoke(input=context)

    return ModelGenerationResponse(text=result.get("text"))



@router.get("/predict/{username}")
async def get_predict(username: str,
                      db: AsyncSession = Depends(get_db_session)) -> List[RetrospectiveResponse]:
    result = await db.execute(select(Retrospective)
                              .filter(Retrospective.username == username)
    )
    retrospectives = result.scalars().all()
    return [
        RetrospectiveResponse(text=retrospective.text)
        for retrospective in retrospectives
    ]



@router.post("/predict")
async def predict(request: RetrospectiveRequest,
                  tf_model: tuple = Depends(get_tf_model),
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
    
    summary = await async_func(make_summary, chunks, device, tokenizer, model)
    
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



@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket,
                             tf_model: tuple = Depends(get_tf_model),
                             llm_chain: LLMChain = Depends(get_llm_chain),
                             db: AsyncSession = Depends(get_db_session)):
    device, tokenizer, model = tf_model
    
    await websocket.accept()
    await websocket.send_text("Connection successful")

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "get_predict":
                result = await db.execute(select(Retrospective)
                              .filter(Retrospective.username == data.get("username"))
                )
                retrospectives = result.scalars().all()
                
                response = [{"text": retrospective.text} for retrospective in retrospectives]
                await websocket.send_json(response)
            
            elif data.get("type") == "post_predict":
                user_inputs = data.get("user_inputs")
                user_inputs_length = len(user_inputs)

                chunks = []
                turns_per_chunk = 5
                total_chunks = user_inputs_length // turns_per_chunk if user_inputs_length % turns_per_chunk == 0 else user_inputs_length // turns_per_chunk + 1
                for i in range(total_chunks):
                    start = i * turns_per_chunk
                    end = (i + 1) * turns_per_chunk if (i + 1) * turns_per_chunk < user_inputs_length else user_inputs_length
                    chunks.append("[BOS]" + "[SEP]".join(user_inputs[start:end]) + "[EOS]")
                
                summary = await async_func(make_summary, chunks, device, tokenizer, model)
                
                context = {
                    "username": data.get("username"),
                    "summary": summary,
                }

                result = await llm_chain.ainvoke(input=context)
                result = result.get("text").strip()

                retrospective = Retrospective(
                    username=data.get("username"),
                    text=result
                )

                db.add(retrospective)
                await db.commit()

                response = {"text": result} #  if reference result from retrospective.text, must await db.refresh(retrospective)
                await websocket.send_json(response)
            
            else:
                await websocket.send_json({"error": "Invalid request type."})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")

    
    



