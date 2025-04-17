import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# Model config
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.9

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant= True
)

@asynccontextmanager
async def imalive(app: FastAPI):
    print("Gimme a sec...")
    app.state.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    app.state.model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map = "auto",
        quantization_config = quant_config,
        torch_dtype = torch.float16
    )

    print("ok let me cook")

    yield

    print("done cooking")

    del app.state.tokenizer
    del app.state.model

    print("bye gang")

app = FastAPI(title = "chat-endpoint", lifespan=imalive)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any]]  # Accept string or complex structures

class ChatCompletionRequest(BaseModel):
    model: str  # OpenAI requires model as a parameter
    messages: List[Message]
    temperature: Optional[float] = TEMPERATURE
    max_tokens: Optional[int] = MAX_NEW_TOKENS
    top_p: Optional[float] = TOP_P
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None  # OpenAI supports stop sequences
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    tools: Optional[List[dict]] = None  # Add tools support
    tool_choice: Optional[Union[str, dict]] = None  # Add tool_choice support


class ChatCompletionsResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: List[dict]
    usage: dict

def format_messages(messages):
    prompt = ""
    
    for message in messages:
        # Handle complex content structures (lists and dictionaries)
        content = message.content
        if isinstance(content, list):
            # If content is a list of dictionaries with 'text' field, extract and join them
            if all(isinstance(item, dict) and "text" in item for item in content):
                content = "\n".join(item["text"] for item in content)
            else:
                # Otherwise convert the list to a string
                content = str(content)
        elif isinstance(content, dict) and "text" in content:
            content = content["text"]
            
        if message.role == "system":
            prompt += f"<|system|>\n{content}\n"
        elif message.role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"

    # Add final assistant turn to prompt completion for Phi-3
    prompt += "<|assistant|>\n"
    
    return prompt

@app.post("/v1/chat/completions",response_model=ChatCompletionsResponse)
async def create_chat_completion(request:ChatCompletionRequest):
    tokenizer = app.state.tokenizer
    model = app.state.model
    try:
        prompt = format_messages(request.messages)

        inputs = tokenizer(prompt, return_tensors = "pt").to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
            )

        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        input_tokens = inputs.input_ids.shape[1]
        output_tokens = outputs.shape[1] - input_tokens
        
        completion_id = f"chatcmpl-{int(time.time())}"
        response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text.strip(),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
