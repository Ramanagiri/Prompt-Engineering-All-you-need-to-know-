from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-5.3",
    messages=[
        {"role": "user", "content": "Extract name and age from: Ravi is 28 years old"}
    ],
    response_format={"type": "json_object"}
)

print(response.choices[0].message.content)
