from openai import OpenAI
client = OpenAI()
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
