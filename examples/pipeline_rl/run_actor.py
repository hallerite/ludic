import logging
import asyncio

from transformers import AutoTokenizer

# Ludic Imports
from ludic.agent import Agent
from ludic.context import FullDialog
from ludic.inference import VLLMChatClient, InferenceSpec, SamplingParams, ReturnSpec, HFChatTemplate
from ludic.parsers import xml_tag_parser
from ludic.training import (
    RolloutEngine,
    run_pipeline_actor,
    EnvSpec,
    ProtocolSpec,
    RolloutRequest,
    make_reinforce,
)
from ludic.interaction import SingleAgentSyncProtocol

# Env Import
from environments.tic_tac_toe import TicTacToeEnv

# ---------------------------------------------------------------------------
# Configuration (Must match Trainer!)
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
REDIS_URL = "redis://localhost:6379/0"
QUEUE_KEY = "ludic_tictactoe_queue"

logging.basicConfig(level=logging.INFO)

def create_engine(client: VLLMChatClient, chat_template: HFChatTemplate) -> RolloutEngine:
    """Setup the Environment and Agent logic."""
    
    # 1. Environment Registry
    env_registry = {"tictactoe": lambda **kwargs: TicTacToeEnv(**kwargs)}
    
    # 2. Protocol Factory
    training_prompt = base_prompt + "\n\nOutput your move as a single XML tag, e.g., <move>A1</move>."

    def create_protocol():
        return SingleAgentSyncProtocol(
            agent=Agent(
                client=client, 
                model=MODEL_NAME, 
                ctx=FullDialog(system_prompt=training_prompt), 
                parser=xml_tag_parser("move"),
                chat_template=chat_template,
            ),
            stop_on_parse_error=True,
        )

    protocol_registry = {"single_agent": create_protocol}
    
    return RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

def make_requests() -> list[RolloutRequest]:
    """Curriculum Definition."""
    base_prompt = TicTacToeEnv().suggested_sysprompt or ""
    training_prompt = base_prompt + "\n\nOutput your move as a single XML tag, e.g., <move>A1</move>."

    inference = InferenceSpec(
        sampling=SamplingParams(temperature=1.0, max_tokens=100),
        return_=ReturnSpec.for_eval(return_token_ids=True),
    )
    
    return [
        RolloutRequest(
            env=EnvSpec(kind="tictactoe", kwargs={"agent_starts": True}),
            protocol=ProtocolSpec(kind="single_agent", kwargs={}),
            inference=inference,
            num_episodes=1, 
        )
    ]

async def main():
    print("🎬 Starting ACTOR Node")
    print(f"   vLLM:  {VLLM_HOST}:{VLLM_PORT}")
    print(f"   Redis: {REDIS_URL}")
    
    # Lightweight client (Inference only, no weight updates)
    client = VLLMChatClient(host=VLLM_HOST, port=VLLM_PORT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    chat_template = HFChatTemplate(tokenizer)
    
    engine = create_engine(client, chat_template)
    credit_assigner = make_reinforce(gamma=0.99).credit_assigner

    # Run Infinite Generation Loop
    await run_pipeline_actor(
        engine=engine,
        requests_fn=make_requests,
        credit_assigner=credit_assigner,
        redis_url=REDIS_URL,
        queue_key=QUEUE_KEY,
        max_steps=5,
        concurrency=16,      # Higher concurrency = higher throughput
        client=client,       # Needed to tag data with policy version
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Actor stopped.")
