import guidance
from typing import Dict

# Create a model instance
model = guidance.llms.LLamaCpp("models/DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf")

# Define a structured prompt with guidance syntax
prompt = guidance("""
You are a helpful assistant. Your task is to rewrite/paraphrase a news 
article journal pretending you are {{persona}}.

Here is the paragraph:
{{article}}

First, let's think through the necessary steps to solve this problem:
<think>
{{#gen 'reasoning'}}{{/gen}}
</think>

Based on my analysis, here's my paragraph:
{{#gen 'response'}}{{/gen}}
""")


def get_response(persona: str, article: str) -> Dict[str, str]:
    result = prompt(
        persona=persona,
        article=article,
        llm=model
    )
    return {
        "reasoning": result["reasoning"],
        "response": result["response"]
    }


if __name__ == "__main__":
    articles = [
        ""
    ]

    persona = "a 50 year old man born in England in 1850"

    for article in articles:
        print(f"\nArticle: {article}")
        result = get_response(persona, article)
        print("\nReasoning:")
        print(result["reasoning"])
        print("\nResponse:")
        print(result["response"])
        print("-" * 50)
