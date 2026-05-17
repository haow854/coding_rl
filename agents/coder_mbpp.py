import os
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer

CODER_SYSTEM_PROMPT = """
You are a Python coding assistant specializing in incremental development.

You are given:
1) A full problem description.
2) Sample testcase.
3) The current state of the Python code.
4) The NEXT SPECIFIC STEP to implement.

Your task:
- Extend the "Current Code" by implementing the "Next Step",
  but respect the requirements of BOTH original problem description and sample testcase.

OUTPUT RULES:
- ONLY output code. NO explanations, markdown notes, or comments.
- The function name and signature MUST match the testcase exactly.
"""

CODER_SYSTEM_PROMPT_NO_INCRE = """
You are a Python coding expert.
"""
class Coder:
    def __init__(self, engine, tokenizer):
        self.engine = engine
        self.tokenizer = tokenizer
        print("coder initialized")

    async def code_vllm(self, plan_list, s):
        current_code = ""
        for step in plan_list:
            messages = [
                {"role": "system", "content": CODER_SYSTEM_PROMPT},
                {"role": "user", "content": f"The problem is as follows.\n{s.text}\n\n"
                                            f"Sample testcase: {s.test_list[0]}\n\n"
                                            f"Current code: {current_code}"
                                            f"Next step: {step}"}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 🚀 关键：这里 await 时，vLLM 会在后台自动把多个并发任务拼成 Batch 跑
            response = await self.vllm_generate(prompt)
            current_code = response.strip()
        return current_code

    async def code_no_incre(self, plan_text, s):
        user_prompt = (
            f"\n{s.prompt}\n\n"
            f"{s.test_list[1]}\n\n"
            f"Here's an implementation plan for your reference: {plan_text}"
        )

        messages = [
            {"role": "system", "content": CODER_SYSTEM_PROMPT_NO_INCRE},
            {"role": "user", "content": user_prompt}
        ]
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        response = await self.vllm_generate(full_prompt)
        code = response.strip()
        return code


    async def vllm_generate(self, prompt: str) -> str:
        sampling_params = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=8192)
        request_id = os.urandom(8).hex()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        final_output = ""
        async for request_output in results_generator:
            final_output = request_output.outputs[0].text
        return final_output

