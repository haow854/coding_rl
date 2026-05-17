import os
from vllm import SamplingParams, TokensPrompt

PLANNER_SYSTEM_PROMPT = """
You are an coding planner for programming problems.

You are given:
1) A full problem description.
2) Sample testcase.

Your task:
- Produce a precise coding plan that a coder model can implement directly.

OUTPUT RULES:
- Provide 1 to 4 steps.
- Output ONLY a JSON list of strings.
- Each string must follow the format: "step X: ..."
- No markdown, no code blocks, no explanations.

Example:
["step 1: ...", "step 2: ..."]


"""


class Planner:
    def __init__(self, engine, tokenizer):
        self.engine = engine
        self.tokenizer = tokenizer
        print("planner initialized")

    async def plan(self, problem):
        user_prompt = f"The problem is as follows.\n{problem}\n"

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = await self.vllm_generate(prompt)
        return response.strip()

    async def vllm_generate(self, prompt: str) -> str:
        sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.9,
            max_tokens=8192
        )
        request_id = os.urandom(8).hex()

        token_ids = self.tokenizer.encode(prompt)
        prompt_input = TokensPrompt(prompt_token_ids=token_ids)

        results_generator = self.engine.generate(prompt_input, sampling_params, request_id)

        final_output = ""
        async for request_output in results_generator:
            final_output = request_output.outputs[0].text

        return final_output