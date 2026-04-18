# Prompt Engineering (Core Skill)
# 🔹 1. Zero-shot Prompting

## 📖 Concept

Zero-shot prompting means giving **no examples**. The model directly answers the query.

---

## 🧾 Prompt

```
Classify the sentiment:

"I love this product!"
```

---

## 💻 LLM Code

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-5.3",
    messages=[
        {"role": "user", "content": "Classify sentiment: I love this product!"}
    ]
)

print(response.choices[0].message.content)
```

---

## ✅ Output

```
Positive
```

---
