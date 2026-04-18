from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-5.3",
    messages=[
        {
            "role": "user",
            "content": "Solve step by step: If 1 pen costs $2, how much for 4 pens?"
        }
    ]
)

print(response.choices[0].message.content)
