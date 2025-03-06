from .package_install import install
import os
import torch

use_api = os.getenv("QUEST_USE_MODEL_API", "false") == "true"


if use_api:
    install("litellm")
    from litellm import completion

    def llm_function(user_prompt, system_prompt=None):
        messages = [
            {"content": user_prompt, "role": "user"}
        ]
        if system_prompt is not None:
            messages.insert(0, {"content": system_prompt, "role": "system"})
        response = completion(model=os.getenv("QUEST_LM_MODEL"), messages=messages)
        text = response.choices[0].message.content
        return text

else:

    os.environ['HF_HOME'] = '/app/cache/hf_home'
    install("transformers")
    install("accelerate")

    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from accelerate.test_utils.testing import get_backend

    class Model:

        def __init__(self, model_name=os.getenv("QUEST_LM_MODEL")):
            self.device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
            self.model_name = model_name
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.generator = pipeline(
                'text-generation',
                model=self.model,
                tokenizer=self.tokenizer
            )

            self.generation_args = {
                "max_length": 1024,
                "truncation": True,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
            }
        
        def get_chat_response(self, query_message:str, token_length:int = 100, system_prompt=None):
            
            # using chat template
            messages = []
            if system_prompt is not None:
                messages.insert(0, {
                    "role": "system",
                    "content": system_prompt,
                })
            messages.append({
                "role": "user",
                "content": query_message,
            })

            obj = self.generator(messages, **self.generation_args)
            return obj[0]['generated_text']

    model = Model()

    def llm_function(user_prompt, system_prompt=None):
        return model.get_chat_response(user_prompt, system_prompt=system_prompt)

