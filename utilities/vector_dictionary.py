from .embedding_models import embed
import torch


def chunk(paragraph, chunk_size=256):
    return [paragraph[i:i + chunk_size] for i in range(0, len(paragraph), chunk_size)]


class Vector_Text_Dictionary:

    def __init__(self, paragraphs, metadata=None, chunk_size=256):
        self.paragraphs = paragraphs
        self.metadata = metadata

        self.embeddings = []
        self.indices = []
        for i, paragraph in enumerate(paragraphs):
            embeddings = embed(chunk(paragraph, chunk_size=chunk_size))
            self.embeddings.append(embeddings)
            self.indices.append(torch.ones(embeddings.shape[0], dtype=torch.int)*i)

        self.embeddings = torch.concat(self.embeddings, dim=0)
        self.indices = torch.concat(self.indices, dim=0)


    def match(self, query, k=1):
        query_embedding = embed(query)
        metric = torch.nn.functional.cosine_similarity(self.embeddings, torch.reshape(query_embedding, [1, -1]), dim=-1)
        # get top_k indices
        top_k_indices = torch.topk(metric, k=k, largest=True, sorted=True).indices
        # get top_k para_indices
        top_k_para_indices = self.indices[top_k_indices[0]]
        # change to python list
        return top_k_para_indices.tolist()
        

    def get_paragraph(self, para_index):
        return self.paragraphs[para_index]
    

    def get_metadata(self, para_index):
        return self.metadata[para_index]