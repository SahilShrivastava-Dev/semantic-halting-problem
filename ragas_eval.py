"""
ragas_eval.py

This script evaluates Retrieval-Augmented Generation (RAG) pipelines using the 
Ragas v0.4 framework. It calculates metrics such as Faithfulness, Answer Relevancy, 
Context Precision, and Context Recall using open-source models hosted on Hugging Face.
"""
import os
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper

# 1. Setup the Judge LLM (Using Zephyr-7b-beta which natively supports Chat format and is ungated)
hf_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=1024,
    do_sample=False
)
hf_chat_model = ChatHuggingFace(llm=hf_endpoint)
ragas_llm = LangchainLLMWrapper(hf_chat_model)

# 2. Setup the Embedding Model
ragas_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# 1. Prepare your data
# In reality, this comes from your LangChain/LlamaIndex pipeline logs
data = {
    "question": ["What is the structural foundation of the new Dubai property?"],
    "answer": ["The foundation of the new Dubai property is built using a reinforced cement base with steel beams."],
    "contexts": [[
        "The new Dubai property features state-of-the-art architecture.",
        "To ensure structural integrity in the sandy terrain, the foundation utilizes a deep reinforced cement base.",
        "Steel beams are used extensively in the upper framework."
    ]],
    "ground_truth": ["The foundation is a reinforced cement base."]
}

dataset=Dataset.from_dict(data)
print(dataset)

metrics_to_run=[
    Faithfulness(),
    AnswerRelevancy(),
    ContextPrecision(),
    ContextRecall()
]

result = evaluate(
    dataset=dataset,
    metrics=metrics_to_run,
    llm=ragas_llm,
    embeddings=ragas_emb
)
print("\n--- Ragas Evaluation Scores ---")
print(result)
    
# You can also convert the detailed row-by-row results to a Pandas DataFrame
df = result.to_pandas()