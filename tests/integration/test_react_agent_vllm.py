from __future__ import annotations

import pytest
from transformers import AutoTokenizer  # Only for verification

from ludic.inference.vllm_client import VLLMChatClient
from ludic.inference import InferenceSpec, SamplingParams, ReturnSpec, HFChatTemplate, HermesToolParser
from ludic.agents.react_agent import ReActAgent
from ludic.context.full_dialog import FullDialog
from tests._mocks import _mock_parser

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def get_secret_code(hint: str) -> str:
    """
    Returns the secret code based on a hint. 
    The agent MUST call this to answer the question.
    """
    if "blue" in hint.lower():
        return "BLUE-42"
    return "RED-99"

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("vllm_server", [{"enable_tools": True}], indirect=True)
async def test_react_agent_vllm_tool_call_loop(
    vllm_client: VLLMChatClient,
    vllm_model_name: str,
) -> None:
    """
    End-to-end integration test:
    1. ReActAgent connects to real vLLM server.
    2. Agent is given a tool `get_secret_code`.
    3. Agent is asked to find the code for 'blue'.
    4. Agent must: Think -> Call Tool -> See 'BLUE-42' -> Output Final Answer.
    """
    
    tokenizer = AutoTokenizer.from_pretrained(vllm_model_name, trust_remote_code=True)
    chat_template = HFChatTemplate(tokenizer, tool_parser=HermesToolParser())

    agent = ReActAgent(
        client=vllm_client,
        model=vllm_model_name,
        ctx=FullDialog(),
        parser=_mock_parser,
        tools=[get_secret_code],
        max_react_steps=3,
        chat_template=chat_template,
    )

    # 2. Setup the Prompt
    system_prompt = (
        "You are a helpful assistant with access to tools. "
        "To find the secret code, you MUST use the tool 'get_secret_code'. "
        "Do not guess. "
        "When you have the answer, output it inside <answer> tags, like <answer>CODE</answer>."
    )
    
    agent.reset(system_prompt=system_prompt)
    
    # 3. Run the Interaction
    agent.on_env_reset("What is the secret code for the blue hint?", {})

    print("\n--- Starting ReAct Loop ---")
    act_result = await agent.act(
        inference=InferenceSpec(
            sampling=SamplingParams(temperature=0.0, max_tokens=256),
            return_=ReturnSpec.for_eval(return_token_ids=True),
        )
    )
    final_step = act_result.final_step
    raw_text = final_step.action
    token_trace = final_step.trace
    print(f"--- Final Output ---\n{raw_text}\n--------------------")

    # 4. Debug: Print Classic Trajectory (High Level)
    print("\n" + "="*60)
    print(" CLASSIC TRAJECTORY (agent._ctx.messages)")
    print("="*60)
    for i, msg in enumerate(agent._ctx.messages):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")
        
        print(f"\n[{i}] {role}:")
        if content:
            print(f"  Content: {content}")
        if tool_calls:
            print(f"  Tool Calls: {tool_calls}")

    # 5. Debug: Print ACTUAL Model Input (Decoded Token IDs)
    print("\n" + "="*60)
    print(" ACTUAL MODEL INPUT (Decoded prompt token IDs)")
    print("="*60)
    
    # The ReActAgent forces return_token_ids=True, so a token trace should be available.
    prompt_ids = token_trace.prompt_token_ids if token_trace is not None else None
    
    if prompt_ids:
        try:
            tokenizer = AutoTokenizer.from_pretrained(vllm_model_name, trust_remote_code=True)
            decoded_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
            print(decoded_text)
            
            # Verification: The tool schema MUST be in this text if the template worked
            assert "get_secret_code" in decoded_text
            assert "function" in decoded_text
            print("\n✅ Verified: Tool definitions were present in the model input.")
            
        except Exception as e:
            print(f"⚠️ Could not decode token IDs: {e}")
    else:
        print("⚠️ No token trace found in response.")

    print("="*60 + "\n")

    # 6. Assertions
    assert "BLUE-42" in raw_text, f"Agent failed to retrieve secret code. Got: {raw_text}"
    
    history = agent._ctx.messages
    tool_calls_found = False
    tool_results_found = False
    
    for msg in history:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc["function"]["name"] == "get_secret_code":
                    tool_calls_found = True
                    print(f"✅ Found Tool Call: {tc}")
        
        if msg.get("role") == "tool" and "BLUE-42" in msg.get("content", ""):
            tool_results_found = True
            print(f"✅ Found Tool Result: {msg['content']}")

    assert tool_calls_found, "Agent never attempted to call 'get_secret_code'"
    assert tool_results_found, "Agent never received the tool output 'BLUE-42'"

    print("🎉 Integration Test Passed: Agent successfully used tools on vLLM.")
