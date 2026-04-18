# 🔹 2. Few-shot Prompting

## 📖 Concept

Few-shot prompting uses **examples to guide the model**.

---

## 🧾 Prompt

```
Classify sentiment:

Text: I hate this movie
Sentiment: Negative

Text: This is amazing
Sentiment: Positive

Text: It is okay
Sentiment:
```

---

## 💻 LLM Code

```python
response = client.chat.completions.create(
    model="gpt-5.3",
    messages=[
        {
            "role": "user",
            "content": """Classify sentiment:

Text: I hate this movie
Sentiment: Negative

Text: This is amazing
Sentiment: Positive

Text: It is okay
Sentiment:"""
        }
    ]
)

print(response.choices[0].message.content)
```

---

## ✅ Output

```
Neutral
```

---
