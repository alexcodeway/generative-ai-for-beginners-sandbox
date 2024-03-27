import ollama

prompt = "Show me 5 recipes for a dish with the following ingredients: chicken, potatoes, and carrots. Per recipe, list all the ingredients used"
completion = ollama.generate(model='llama2', prompt=prompt)

print(completion['response'])

prompt = "Please remove recipes with garlic as I'm allergic and replace it with something else. Also, please produce a shopping list for the recipes, considering I already have chicken, potatoes and carrots at home."
completion = ollama.generate(model='llama2', prompt=prompt)

print(completion['response'])