from ollama import chat

role = 'user'

print('Chat started')

while True:
    prompt = input('>>')
    messages = [{'role': role,'content': prompt,},]
    response = chat('mistral', messages=messages)
    print(response['message']['content'])