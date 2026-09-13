from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

# ✅ Step 1: Create HuggingFace LLM
hf_llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    max_new_tokens=200,
)

# ✅ Step 2: Wrap it as a Chat Model
llm = ChatHuggingFace(llm=hf_llm)

# ✅ Step 3: Invoke
response = llm.invoke("What is the capital of India?")
print(response.content)
