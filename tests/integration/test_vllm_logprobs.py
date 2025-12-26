from __future__ import annotations

import gc
import time

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ludic.inference import HFChatTemplate
from ludic.inference.vllm_client import VLLMChatClient
from ludic.inference.request import TokenCompletionRequest, ReturnSpec
from ludic.inference.sampling import SamplingParams

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.diagnostic]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vllm_server",
    [
        {"dtype": "float16"},
        {"dtype": "bfloat16"},
        {"dtype": "float32"},
    ],
    indirect=True,
)
async def test_vllm_client_returns_logprobs(
    vllm_client: VLLMChatClient,
    vllm_server,
    vllm_model_name: str,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """
    Ensure vLLM returns per-token logprobs when requested (logprobs=1 by default).
    """
    # Use "no-op" sampling settings so HF teacher-forcing logprobs match the
    # distribution vLLM should report (no temperature scaling / truncation).
    sampling = SamplingParams(
        temperature=1.0,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        vllm_model_name,
        trust_remote_code=True,
    )
    chat_template = HFChatTemplate(tokenizer)

    long_sampling = SamplingParams(
        temperature=1.0,
        max_tokens=max(256, 32),
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None,
    )

    scenarios = [
        (
            "short_single",
            [
                {"role": "system", "content": "You are a test assistant. Answer concisely."},
                {"role": "user", "content": "Say the word 'ping'."},
            ],
            sampling,
        ),
        (
            "long_sequence",
            [
                {"role": "system", "content": "You are a test assistant. Answer concisely."},
                {"role": "user", "content": "Respond with exactly: ping pong pang pung peng ping pong pang pung peng."},
            ],
            long_sampling,
        ),
    ]

    # (scenario_name, sampling_cfg, seed, prompt_token_ids, completion_token_ids, completion_logprobs)
    responses: list[tuple[str, SamplingParams, int, list[int], list[int], list[float]]] = []

    for scenario_name, messages, sampling_cfg in scenarios:
        seed = 0
        prompt_token_ids = chat_template.apply(
            messages,
            add_generation_prompt=True,
        ).prompt_token_ids
        resp, _ = await vllm_client.complete_tokens(
            TokenCompletionRequest(
                model=vllm_model_name,
                prompt_token_ids=prompt_token_ids,
                sampling=sampling_cfg,
                seed=seed,
                return_=ReturnSpec.for_rl(top_logprobs_k=1),
            )
        )

        assert resp.text.strip() != ""
        assert resp.completion_token_ids is not None
        assert resp.completion_logprobs is not None
        assert len(resp.completion_logprobs) == len(resp.completion_token_ids)

        # Sanity-check that the returned token IDs are consistent with the returned text.
        completion_ids = list(resp.completion_token_ids)
        reencoded = tokenizer.encode(resp.text, add_special_tokens=False)
        if (
            tokenizer.eos_token_id is not None
            and completion_ids
            and completion_ids[-1] == tokenizer.eos_token_id
        ):
            assert reencoded == completion_ids[:-1]
        else:
            assert reencoded == completion_ids

        prompt_ids = resp.prompt_token_ids
        assert prompt_ids is not None

        responses.append(
            (
                scenario_name,
                sampling_cfg,
                seed,
                list(prompt_ids),
                list(resp.completion_token_ids),
                list(resp.completion_logprobs),
            )
        )

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for a GPU-to-GPU logprob cross-check.")

    # Cross-check by teacher-forcing the same model locally on GPU in multiple precisions.
    # Stop vLLM first so the GPU has room for the HF model.
    vllm_server.stop()
    time.sleep(1.0)

    def teacher_force_logprobs(prompt_ids: list[int], completion_ids: list[int], dtype: torch.dtype) -> list[float]:
        gc.collect()
        torch.cuda.empty_cache()

        device = torch.device("cuda")
        model = None
        try:
            model = AutoModelForCausalLM.from_pretrained(
                vllm_model_name,
                dtype=dtype,
                trust_remote_code=True,
            ).to(device)
            model.eval()

            combined = prompt_ids + completion_ids
            input_ids = torch.tensor(combined, dtype=torch.long, device=device).unsqueeze(0)
            attn = torch.ones_like(input_ids)

            with torch.inference_mode():
                logits = model(input_ids=input_ids, attention_mask=attn).logits
                token_logp = F.log_softmax(logits.float()[:, :-1, :], dim=-1)  # [1, T-1, V]
                targets = input_ids[:, 1:]  # [1, T-1]
                gathered = torch.gather(token_logp, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [1, T-1]

            prompt_len = len(prompt_ids)
            completion_mask = torch.zeros_like(gathered, dtype=torch.bool)
            completion_mask[:, prompt_len - 1 :] = True  # predictions corresponding to completion tokens
            return gathered[completion_mask].cpu().tolist()
        finally:
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()

    try:
        server_dtype = getattr(vllm_server, "dtype", "bfloat16")
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        rel_tol = 1e-2
        abs_tol = 0.1
        scenario_summaries: list[tuple[str, int, list[tuple[str, float, float, bool]]]] = []

        target_dtype = dtype_map.get(server_dtype)
        if target_dtype is None:
            raise RuntimeError(f"Unsupported dtype requested for server: {server_dtype}")

        for scenario_name, _sampling_cfg, _seed, prompt_ids, completion_token_ids, completion_logprobs in responses:
            scenario_results: list[tuple[str, float, float, bool]] = []

            hf_logprobs = teacher_force_logprobs(prompt_ids, completion_token_ids, target_dtype)

            assert len(hf_logprobs) == len(completion_logprobs)

            abs_err = [abs(g - e) for g, e in zip(hf_logprobs, completion_logprobs)]
            max_abs_err = max(abs_err)
            mean_abs_err = sum(abs_err) / len(abs_err)
            within_tolerance = all(
                abs(g - e) <= max(abs(e) * rel_tol, abs_tol) for g, e in zip(hf_logprobs, completion_logprobs)
            )
            scenario_results.append((server_dtype, max_abs_err, mean_abs_err, within_tolerance))

            scenario_summaries.append(
                (scenario_name, len(completion_token_ids), scenario_results)
            )

        with capfd.disabled():
            print(f"Precision comparison against vLLM completion logprobs (server dtype={server_dtype}):")
            for scenario_name, completion_len, scenario_results in scenario_summaries:
                print(f"Scenario '{scenario_name}' (completion tokens={completion_len}):")
                for name, max_err, mean_err, ok in scenario_results:
                    print(
                        f"  - {name}: max abs err={max_err:.4f}, mean abs err={mean_err:.4f}, status="
                        f"{'within tolerance' if ok else f'EXCEEDS tolerance (rel={rel_tol}, abs={abs_tol})'}"
                    )
                offenders = [name for (name, _, _, ok) in scenario_results if not ok]
                if offenders:
                    print(
                        "    ! Observed logprob mismatch beyond tolerance for: "
                        + ", ".join(offenders)
                    )
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        vllm_server.start()
