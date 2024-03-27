from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.chains import create_extraction_chain

# taken from: https://python.langchain.com/docs/integrations/chat/ollama_functions

model = OllamaFunctions(model="mistral")

# weather
model = model.bind(
    functions=[
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, " "e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        }
    ],
    function_call={"name": "get_current_weather"},
)

weather = model.invoke("what is the weather in Boston?")

print(weather)

# extractions
# Schema
schema = {
    "properties": {
        "name": {"type": "string"},
        "height": {"type": "integer"},
        "hair_color": {"type": "string"},
    },
    "required": ["name", "height"],
}

# Input
input = """Alex is 5 feet tall. Claudia is 1 feet taller than Alex and jumps higher than him. Claudia is a brunette and Alex is blonde."""

# Run chain
chain = create_extraction_chain(schema, model)
output = chain.run(input)

print(output)