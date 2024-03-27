import ollama
import json

model = 'llama2'

student_1_description="Emily Johnson is a sophomore majoring in computer science at Duke University. She has a 3.7 GPA. Emily is an active member of the university's Chess Club and Debate Team. She hopes to pursue a career in software engineering after graduating."
 
student_2_description = "Michael Lee is a sophomore majoring in computer science at Stanford University. He has a 3.8 GPA. Michael is known for his programming skills and is an active member of the university's Robotics Club. He hopes to pursue a career in artificial intelligence after finshing his studies."

prompt1 = f'''
Please extract the following information from the given text and return it as a JSON object:

name
major
school
grades
club

This is the body of text to extract the information from:
{student_1_description}
'''


prompt2 = f'''
Please extract the following information from the given text and return it as a JSON object:

name
major
school
grades
club

This is the body of text to extract the information from:
{student_2_description}
'''

ollama_response1 =  ollama.generate(model=model, prompt=prompt1, format='json')['response']
ollama_response2 =  ollama.generate(model=model, prompt=prompt2, format='json')['response']

print(ollama_response1)
print(ollama_response2)

# Loading the response as a JSON object
json_response1 = json.loads(ollama_response1)
json_response2 = json.loads(ollama_response2)