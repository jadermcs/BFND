import guidance
from guidance import gen, models
from datasets import load_dataset, Dataset
import random
from typing import Dict


model = models.LlamaCpp(
    "models/DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
    flash_attn=True,
    echo=False,
)


@guidance
def get_response(lm, article: str) -> Dict[str, str]:
    lm += f"""
You are a person that misunderstand important facts. Your task is to edit
small facts of a news article to alter the information, making it 
different from the original.
First example:
Original article: "The unemployment rate increased from previous year by 2%."
Altered article: "The unemployment rate increased from previous year by 7%."
Second example:
Original article: "Brazil attacked Bosnia last Thursday."
Altered article: "Bosnia attacked Brazil last Thursday."

Here is the paragraph:
{article}

First, let's think through the necessary steps to solve this problem:
<think>
{gen('reasoning', stop='</think>')}
</think>

Based on my analysis, here's my paragraph:
{gen('response')}
"""
    return lm


if __name__ == "__main__":

    data = load_dataset("Pravincoder/CNN_News")["train"].shuffle(seed=42)

    new_data = []
    count = 0
    for article in data:
        if len(article['highlights']) > 1000:
            continue
        count += 1
        fake = False
        if random.random() > 0.5:
            fake = True
            print(f"\nOriginal article: {article['highlights']}")
            result = model + get_response(article['highlights'])
            print("\nReasoning:")
            print(result["reasoning"])
            print("\nResponse:")
            print(result["response"])
            print("-" * 50)
        else:
            result = {"response": article['highlights']}
        new_data.append({
            "original_article": article["text"],
            "highlights": result["response"],
            "fake_news": fake,
            })
        if count > 10:
            break
    new_data = Dataset.from_list(new_data)
    new_data = new_data.train_test_split(test_size=0.3, seed=42)
    new_data["train"].to_json("fake_news.train.json")
    new_data["test"].to_json("fake_news.test.json")
