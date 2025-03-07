from .package_install import install
import os
import torch

lm_deployment_type = os.getenv("QUEST_EMBEDDING_DEPLOYMENT", "cloud-api-litellm")

if lm_deployment_type == "cloud-api-litellm":
    install("litellm")
    import litellm
    from litellm import embedding

    cloud_endpoint = os.getenv("CLOUD_ENDPOINT", None)
    cloud_api_key = os.getenv("CLOUD_API_KEY", None)

    if cloud_endpoint is not None:
        litellm.api_base = cloud_endpoint
    if cloud_api_key is not None:
        litellm.api_key = cloud_api_key

    model = os.getenv("QUEST_EMBEDDING_MODEL")
        
    def embed(text):
        # if a list then embed list
        if isinstance(text, list):
            response = embedding(
                model=model, 
                input=text
            )
            return torch.tensor([embedding["embedding"] for embedding in response.data], dtype=torch.float32)
        else:
            response = embedding(
                model=model, 
                input=[text]
            )
            return torch.tensor(response.data[0]["embedding"], dtype=torch.float32)

    



