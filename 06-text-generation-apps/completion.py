import ollama

# simple completion
prompt = "Complete the following: Once upon a time there was a"
# Generate is ollama's term for a completion
completion = ollama.generate(model='llama2', prompt=prompt)

print(completion['response'])