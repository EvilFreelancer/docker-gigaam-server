from openai import OpenAI

client = OpenAI(base_url='http://127.0.0.1:5000', api_key='<key>')

file = open("NHUg0pdEXyg.opus", "rb")
model = 'rnnt'

test = client.audio.transcriptions.create(
    model=model,
    file=file,
)
print(test)
