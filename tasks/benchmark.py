import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time
import torch

print("Torch version:", torch.__version__)
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print("Device:", device)

def selu(x, alpha:float=1.67, lmbda:float=1.05):
    return lmbda * torch.where(x > 0, x, alpha * torch.exp(x) - alpha)

x = torch.randn(1_000_000, device=device)

start_time = time.time()
for i in range(10000):
    selu(x)
print(f"SELUs took {time.time() - start_time:.4f} seconds")

# test torch legacy jit.script

selu_jit = torch.jit.script(selu)

start_time = time.time()
for i in range(10000):
    selu_jit(x)
print(f"JITted SELUs took {time.time() - start_time:.4f} seconds")

# test torch jit with more complex function

def loop_selu(x):
    sum = torch.zeros(1_000_000, device=x.device)
    for i in range(1000):
        sum = sum + selu(x)
    return torch.sum(sum)

loop_selu_jit = torch.jit.script(loop_selu)

start_time = time.time()
loop_selu_jit(x)
print(f"JITted loop SELUs took {time.time() - start_time:.4f} seconds")


# test torch compile

loop_selu_compiled = torch.compile(loop_selu)
loop_selu_compiled(x) # compiles on first call

start_time = time.time()
loop_selu_compiled(x)
print(f"Compiled loop SELUs took {time.time() - start_time:.4f} seconds")


from utilities.language_models import Chat, Chat_Message, Language_Model

lm = Language_Model()
start_stamp = time.time()
text = lm.complete_text("Who's the greatest scientist of all time?")
end_stamp = time.time()
print(text)
print(f"Prompt completion time: {end_stamp - start_stamp:.4f} seconds")


chat = Chat(messages=[
    Chat_Message(role="system", content="You're a historian."),
    Chat_Message(role="user", content="Who is the greatest mathematician?"),
])

start_stamp = time.time()
text = lm.complete_chat(chat)
end_stamp = time.time()
print(text)
print(f"Chat completion time: {end_stamp - start_stamp:.4f} seconds")


from utilities.embedding_models import embed
from utilities.vector_dictionary import Vector_Text_Dictionary

start_stamp = time.time()
pivots = Vector_Text_Dictionary(paragraphs=["Homo sapien", "Canis lupus familiaris", "Felis catus"])
top_k_indices = pivots.match("Dog", k=1)
end_stamp = time.time()
print(pivots.get_paragraph(top_k_indices))
print(f"Retrieval time: {end_stamp - start_stamp:.4f} seconds")