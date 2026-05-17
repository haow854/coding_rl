import os
from vllm import SamplingParams

CODER_SYSTEM_PROMPT_NO_INCRE = """
You are a Python coding expert.
"""


class Coder:
    def __init__(self, engine, tokenizer):
        self.engine = engine
        self.tokenizer = tokenizer
        print("coder initialized")

    async def code_no_incre(self, plan_text, s):
        user_prompt = (
            f"{s.prompt}\n\n"
            f"Here's an implementation plan for your reference:\n{plan_text}"
        )

        messages = [
            {"role": "system", "content": CODER_SYSTEM_PROMPT_NO_INCRE},
            {"role": "user", "content": user_prompt}
        ]

        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = await self.vllm_generate(full_prompt)
        return response.strip()

    async def vllm_generate(self, prompt: str) -> str:
        sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.9,
            max_tokens=8192
        )

        request_id = os.urandom(8).hex()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        final_output = ""
        async for request_output in results_generator:
            final_output = request_output.outputs[0].text

        return final_output