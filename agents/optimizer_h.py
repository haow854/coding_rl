import os
from vllm import AsyncLLMEngine, SamplingParams, TokensPrompt

OPTIMIZER_SYSTEM_PROMPT = """You are a Python code reviewer and optimizer.

Given a problem, a sample testcase, and a code attempt, you must:
Step 1: Judge whether the code is correct or buggy.
Step 2: If buggy, provide a fixed version.
        If correct, return the code unchanged.

Rules:
- Keep the function name and signature exactly as in the testcase.
- Do NOT add test code or assertions.
"""


class Optimizer:
    def __init__(self, engine, tokenizer):
        self.engine = engine
        self.tokenizer = tokenizer
        print("optimizer initialized")

    async def optimize(self, problem, code):
        user_prompt = (
            f"Problem: {problem}\n\n"
            f"Code:\n```python\n{code}\n```"
        )

        messages = [
            {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = await self.vllm_generate(prompt)
        return response.strip()

    async def vllm_generate(self, prompt: str) -> str:
        sampling_params = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=8192)
        request_id = os.urandom(8).hex()
        token_ids = self.tokenizer.encode(prompt)
        prompt_input = TokensPrompt(prompt_token_ids=token_ids)
        results_generator = self.engine.generate(prompt_input, sampling_params, request_id)

        final_output = ""
        async for request_output in results_generator:
            final_output = request_output.outputs[0].text
        return final_output