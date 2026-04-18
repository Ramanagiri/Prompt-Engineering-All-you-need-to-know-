from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-5.3",
    messages=[
        {"role": "user", "content": "Classify sentiment: I love this product!"}
    ]
)

print(response.choices[0].message.content)
