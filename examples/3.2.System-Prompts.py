from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-5.3",
    messages=[
        {"role": "system", "content": "You are a strict math teacher."},
        {"role": "user", "content": "What is 10 / 2?"}
    ]
)

print(response.choices[0].message.content)
