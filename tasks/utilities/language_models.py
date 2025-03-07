from .package_install import install
import os
import torch
from pydantic import BaseModel
from typing import List


class Chat_Message(BaseModel):
    role: str
    content: str

class Chat(BaseModel):
    messages: List[Chat_Message]

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


lm_deployment_type = os.getenv("QUEST_LM_DEPLOYMENT", "cloud-api-litellm")

if lm_deployment_type == "cloud-api-litellm":
    install("litellm")
    import litellm
    from litellm import text_completion, completion

    cloud_endpoint = os.getenv("CLOUD_ENDPOINT", None)
    cloud_api_key = os.getenv("CLOUD_API_KEY", None)

    if cloud_endpoint is not None:
        litellm.api_base = cloud_endpoint
    if cloud_api_key is not None:
        litellm.api_key = cloud_api_key

    def complete_text(user_prompt):
        response = text_completion(model=os.getenv("QUEST_LM_MODEL"), prompt=user_prompt)
        text = response.choices[0].text
        return text
    
    def complete_chat(chat: Chat):
        messages = chat.serialize()
        response = completion(model=os.getenv("QUEST_LM_MODEL"), messages=messages)
        text = response.choices[0].message.content

elif lm_deployment_type == "cloud-api-raw":
    install("requests")
    import requests

    cloud_endpoint = os.getenv("CLOUD_ENDPOINT", None)
    cloud_api_key = os.getenv("CLOUD_API_KEY", None)

    def complete_text(user_prompt):
        response = requests.post(cloud_endpoint, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cloud_api_key}"
        }, json={"input": {"prompt": user_prompt}})
        text = response.json()['output'][0]['choices'][0]['tokens'][0]
        return text
    
    def complete_chat(chat: Chat):
        messages = chat.serialize()
        response = requests.post(cloud_endpoint, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cloud_api_key}"
        }, json={"input": {"messages": messages}})
        text = response.json()['output'][0]['choices'][0]['tokens'][0]
        return text

elif lm_deployment_type == "local-hf":

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
        
        def get_chat_response(self, chat: Chat, token_length:int = 100):
            messages = chat.serialize()
            obj = self.generator(messages, **self.generation_args)
            return obj[0]['generated_text'][-1]['content']
        

        def get_text_completion(self, user_prompt:str, token_length:int = 100):
            return self.generator(user_prompt, max_length=token_length, **self.generation_args)[0]['generated_text']

    model = Model()

    def complete_text(user_prompt):
        return model.get_text_completion(user_prompt)
    
    def complete_chat(chat: Chat):
        return model.get_chat_response(chat)

