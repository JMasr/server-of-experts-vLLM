import os

from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from server_experts.config import logger, MODELS_DIR
from server_experts.models.hf_utils import download_hugging_face_model

# --- Configuration ---
# Hugging Face Token
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

# If your model requires a specific trust_remote_code value
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "True").lower() == "true"

# GPU memory utilization (0.0 to 1.0)
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", 0.90))

# Tensor parallelism size (number of GPUs to use for the model)
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", 1))

# Model parameters
MODEL_ID = "meta-llama/Llama-3.2-1B"
MODEL_NAME_OR_PATH = MODELS_DIR / f"{MODEL_ID.split('/')[0]}" / f"{MODEL_ID.split('/')[-1]}"

if not MODEL_NAME_OR_PATH.exists():
    download_hugging_face_model(model_path=MODEL_NAME_OR_PATH, model_id=MODEL_ID, hf_token=HF_API_TOKEN)

# Experts Section
BASE_MODEL_NICKNAME = MODEL_NAME_OR_PATH.split("/")[-1].lower().replace("-instruct", "").replace("-chat",
                                                                                                 "")  # e.g., "llama-3-8b"

# Definition of experts and their parameters
EXPERT_PERSONAS = {
    "expert-writer": {
        "system_prompt": "You are an expert writer, skilled in crafting compelling narratives, clear explanations, and engaging content. Focus on clarity, tone, and impactful language.",
        "sampling_params": SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
    },
    "expert-reader": {
        "system_prompt": "You are an expert reader and summarizer. Your task is to carefully analyze the provided text, identify key information, and provide concise, accurate summaries or answer questions based on it.",
        "sampling_params": SamplingParams(temperature=0.5, top_p=0.8, max_tokens=300)
    },
    "code-generator": {
        "system_prompt": "You are an expert code generator. Produce clean, efficient, and well-commented code in the requested programming language. If no language is specified, default to Python.",
        "sampling_params": SamplingParams(temperature=0.3, top_p=0.9, max_tokens=1024, stop=["\n```\n"])
    },
    "general-assistant": {
        "system_prompt": "You are a helpful general-purpose AI assistant. Provide informative and concise answers to a wide range of queries.",
        "sampling_params": SamplingParams(temperature=0.6, top_p=0.9, max_tokens=512)
    }
}


# --- Pydantic Models for Request and Response ---
class LLMRequest(BaseModel):
    prompt: str
    # You can add more parameters here if needed, e.g., overriding sampling_params
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    # Add other vLLM sampling params as needed


class LLMResponse(BaseModel):
    request_id: str
    generated_text: str
    model_name: str
    expert_name: str


app = FastAPI(title="Llama Models Server with vLLM", version="1.0.0")

# Global vLLM engine instance
llm_engine: AsyncLLMEngine | None = None

# --- vLLM Engine Initialization ---
@app.on_event("startup")
async def startup_event():
    """
    Initialize the AsyncLLMEngine on application startup.
    """
    global llm_engine

    # Log engine initialization details
    init_message = (
        f"Initializing vLLM engine for model: {MODEL_NAME_OR_PATH}, "
        f"Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}, "
        f"GPU Memory Utilization: {GPU_MEMORY_UTILIZATION}, "
        f"Trust Remote Code: {TRUST_REMOTE_CODE}"
    )
    logger.info(init_message)


    engine_args = AsyncEngineArgs(
        model=MODEL_NAME_OR_PATH,
        tokenizer=MODEL_NAME_OR_PATH,  # Can be same as model, or a specific tokenizer path
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        trust_remote_code=TRUST_REMOTE_CODE,
        # Add other engine arguments as needed, e.g.:
        # dtype="auto", # or "half", "bfloat16"
        # max_model_len=None, # Defaults to model's max length or 2048
        # quantization=None, # e.g., "awq", "gptq"
        # enforce_eager=False,
    )

    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info(f"vLLM engine for {MODEL_NAME_OR_PATH} initialized successfully.")

    # Perform a dummy generation to verify that the model is loaded
    await _perform_dummy_generation()


async def _perform_dummy_generation(llm_engine_to_test: AsyncLLMEngine):
    """
    Perform a dummy generation to ensure the model is loaded.
    """
    try:
        test_params = SamplingParams(max_tokens=10)
        await llm_engine_to_test.generate("Hello, world!", test_params, random_uuid())
        logger.info("Dummy generation successful. Model is loaded.")
    except Exception as error:
        error_message = f"Error during dummy generation: {error}"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=f"Failed to load model for dummy generation: {error}")


# --- API Endpoints ---
# Create a router for our model experts
# The prefix will be /api/v1/{base_model_nickname}
api_router = APIRouter(prefix=f"/api/v1/{BASE_MODEL_NICKNAME}")


@api_router.post("/{expert_name}/invoke", response_model=LLMResponse)
async def invoke_expert(expert_name: str, request: LLMRequest):
    """
    Endpoint to interact with a specific expert persona.
    The `expert_name` is part of the path.
    """
    global llm_engine
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="vLLM engine is not initialized.")

    if expert_name not in EXPERT_PERSONAS:
        raise HTTPException(status_code=404, detail=f"Expert persona '{expert_name}' not found.")

    expert_config = EXPERT_PERSONAS[expert_name]
    system_prompt = expert_config["system_prompt"]

    # Combine system prompt with user prompt
    # For chat models, the format might be more specific (e.g., using roles)
    # This is a simple concatenation for instruction-tuned models.
    # Adjust if your model expects a specific chat template format.
    # For Llama 3 Instruct, a common format is:
    # <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    # {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    # {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    # Simple approach (adjust for specific model chat templates if needed):
    # full_prompt = f"System: {system_prompt}\nUser: {request.prompt}\nAssistant:"

    # Llama 3 Instruct specific templating (example)
    # Note: vLLM's tokenizer might handle chat templates automatically if the model has one configured.
    # If so, you might be able to pass a list of messages.
    # For direct string prompting with Llama 3 Instruct:
    full_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{request.prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    # Get sampling parameters for the expert, allow overrides from request
    sampling_params_dict = expert_config["sampling_params"].to_dict()
    if request.max_tokens is not None:
        sampling_params_dict["max_tokens"] = request.max_tokens
    if request.temperature is not None:
        sampling_params_dict["temperature"] = request.temperature
    if request.top_p is not None:
        sampling_params_dict["top_p"] = request.top_p

    current_sampling_params = SamplingParams(**sampling_params_dict)

    request_id = f"req-{random_uuid()}"

    print(f"--- Request ID: {request_id} ---")
    print(f"Expert: {expert_name}")
    print(f"User Prompt (first 100 chars): {request.prompt[:100]}...")
    # print(f"Full Prompt (first 200 chars): {full_prompt[:200]}...") # Be careful logging full prompts with sensitive data
    print(f"Sampling Params: {current_sampling_params}")
    print("-------------------------------")

    try:
        results_generator = llm_engine.generate(full_prompt, current_sampling_params, request_id)
        final_output = None
        async for request_output in results_generator:
            # For non-streaming, we usually get one result.
            # If streaming, you'd handle multiple partial results here.
            final_output = request_output

        if final_output is None or not final_output.outputs:
            raise HTTPException(status_code=500, detail="LLM generation failed to produce output.")

        generated_text = final_output.outputs[0].text

        # Clean up potential self-prompting if the model repeats the input
        # This is a basic cleanup, might need refinement
        if generated_text.strip().startswith(request.prompt.strip()):
            generated_text = generated_text.strip()[len(request.prompt.strip()):].strip()

        # Some models might include the system prompt or parts of the template in the output.
        # This is a simple way to remove the prompt part from the generated text.
        # More sophisticated chat template handling by vLLM might make this unnecessary.
        # if full_prompt.endswith(generated_text): # This check is too simple
        #    generated_text = generated_text[len(full_prompt):]

        return LLMResponse(
            request_id=request_id,
            generated_text=generated_text.strip(),
            model_name=MODEL_NAME_OR_PATH,
            expert_name=expert_name
        )

    except Exception as e:
        print(f"Error during LLM generation for request {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"LLM generation error: {str(e)}")


# Include the router in the main application
app.include_router(api_router)


@app.get("/", summary="Root endpoint to check server status.")
async def root():
    return {
        "message": f"Llama Model Server with vLLM is running."
                   f" Base model: {MODEL_NAME_OR_PATH}."
                   f" Access experts at /api/v1/{BASE_MODEL_NICKNAME}/{{expert_name}}/invoke"
    }
