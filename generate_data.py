import guidance
from guidance import gen, models
from datasets import load_dataset
from typing import Dict


# Create a model instance
model = models.LlamaCpp("models/DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf",
            n_gpu_layers=-1,
            n_ctx=1024,
            flash_attn=True,
            echo=False)

# Define a structured prompt with guidance syntax

@guidance
def get_response(lm, article: str) -> Dict[str, str]:
    lm += f"""
You are a helpful assistant. Your task is to rewrite/paraphrase small parts of 
a news article to alter meaningfully what is said. Example:
Original article: "The unemployment rate increased from previous year by 2%."
Altered article: "The unemployment rate increased from previous year by 7%."

Here is the paragraph:
{article}

First, let's think through the necessary steps to solve this problem:
<think>
{gen('reasoning')}
</think>

Based on my analysis, here's my paragraph:
{gen('response')}
"""
    return lm


if __name__ == "__main__":

    data = load_dataset("Pravincoder/CNN_News")["train"]

    count = 0
    for article in data:
        print(article)
        count += 1
        print(f"\nArticle: {article['highlights']}")
        result = model + get_response(article['highlights'])
        print("\nReasoning:")
        print(result["reasoning"])
        print("\nResponse:")
        print(result["response"])
        print("-" * 50)
        if count > 3:
            break
