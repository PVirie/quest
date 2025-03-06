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
    install("transformers>=4.45.1")
    install("bitsandbytes>=0.39.0")
    install("accelerate")

    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from accelerate.test_utils.testing import get_backend

    class Model:

        def __init__(self, model_name=os.getenv("QUEST_LM_MODEL")):
            self.model_name = model_name
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
            self.device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Most LLMs don't have a pad token by default
        
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
            model_inputs = self.tokenizer.apply_chat_template(query_message, add_generation_prompt=True, return_tensors='pt').to(self.device)
            input_length = model_inputs.shape[1]
            generated_ids = self.model.generate(model_inputs, do_sample=True, max_new_tokens=token_length)

            return self.tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]

    model = Model()

    def llm_function(user_prompt, system_prompt=None):
        return model.get_chat_response(user_prompt, system_prompt=system_prompt)

