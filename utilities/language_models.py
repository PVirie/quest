from .package_install import install
import os
import torch
from pydantic import BaseModel
from typing import List


class Chat_Message(BaseModel):
    role: str
    content: str

class Chat(BaseModel):
    messages: List[Chat_Message] = []

    def __add__(self, message: Chat_Message):
        concatenated_messages = self.messages + [message]
        return Chat(messages=concatenated_messages)

    def insert(self, index: int, message: Chat_Message):
        self.messages.insert(index, message)

    def append(self, message: Chat_Message):
        self.messages.append(message)

    def serialize(self):
        return [{"role": message.role, "content": message.content} for message in self.messages]
    
    @staticmethod
    def deserialize(json_obj):
        return Chat(messages=[Chat_Message(**message) for message in json_obj["messages"]])


deployment_type = os.getenv("QUEST_LM_DEPLOYMENT", "cloud-api-litellm")

if deployment_type == "cloud-api-litellm":
    install("litellm")
    import litellm
    from litellm import text_completion, completion, embedding

    cloud_endpoint = os.getenv("CLOUD_ENDPOINT", None)
    cloud_api_key = os.getenv("CLOUD_API_KEY", None)

    if cloud_endpoint is not None:
        litellm.api_base = cloud_endpoint
    if cloud_api_key is not None:
        litellm.api_key = cloud_api_key

    class Language_Model:
        def __init__(self, max_length=1024, top_p=0.95, temperature=0.6):
            self.model = os.getenv("QUEST_LM_MODEL")
            self.max_length = max_length
            self.top_p = top_p
            self.temperature = temperature

        def complete_text(self, user_prompt):
            response = text_completion(
                model=self.model, 
                prompt=user_prompt,
                max_tokens=self.max_length,
                top_p=self.top_p,
                temperature=self.temperature,
                stop=["\n"]
            )
            text = response.choices[0].text
            return text
        
        def complete_chat(self, chat: Chat):
            messages = chat.serialize()
            response = completion(
                model=self.model, 
                messages=messages,
                max_tokens=self.max_length,
                top_p=self.top_p,
                temperature=self.temperature
            )
            text = response.choices[0].message.content
            return text

elif deployment_type == "cloud-api-raw":
    install("requests")
    import requests

    cloud_endpoint = os.getenv("CLOUD_ENDPOINT", None)
    cloud_api_key = os.getenv("CLOUD_API_KEY", None)

    class Language_Model:
        def __init__(self, max_length=1024, top_p=0.95, temperature=0.6):
            self.sampling_params = {
                "max_tokens": max_length,
                "top_p": top_p,
                "temperature": temperature,
                "stop": ["\n"]
            }

        def complete_text(self, user_prompt):
            response = requests.post(cloud_endpoint, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cloud_api_key}"
            }, json={
                "input": {"prompt": user_prompt},
                "sampling_params": self.sampling_params
            })
            text = response.json()['output'][0]['choices'][0]['tokens'][0]
            return text
        
        def complete_chat(self, chat: Chat):
            messages = chat.serialize()
            response = requests.post(cloud_endpoint, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cloud_api_key}"
            }, json={
                "input": {"messages": messages},
                "sampling_params": self.sampling_params
            })
            text = response.json()['output'][0]['choices'][0]['tokens'][0]
            return text

elif deployment_type == "local-hf":

    os.environ['HF_HOME'] = '/app/cache/hf_home'
    install("transformers")

    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    class Language_Model:

        def __init__(self, max_length=1024, top_p=0.95, temperature=0.6):
            self.model_name = os.getenv("QUEST_LM_MODEL")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                # torch_dtype=torch.float16,
                # use_flash_attention_2=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.generator = pipeline(
                'text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                device=device
            )

            self.generation_args = {
                "max_new_tokens": max_length,
                "temperature": temperature + 0.01,
                "top_p": top_p,
                "truncation": True,
                "do_sample": True,
                "stop_strings": ["\n"],
                "tokenizer": self.tokenizer,
                "return_full_text": False
            }
        
        def complete_chat(self, chat: Chat):
            messages = chat.serialize()
            obj = self.generator(messages, **self.generation_args)
            return obj[0]['generated_text'][-1]['content']
        
        def complete_text(self, user_prompt:str):
            return self.generator(user_prompt, **self.generation_args)[0]['generated_text']


