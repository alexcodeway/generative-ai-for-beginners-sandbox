# Used to transform the embedding_index_3m.json file to use nomic-embed-text model rather than text-embedding-ada-002
# as Ollama currently does not support this model.

import json
import ollama

model = 'nomic-embed-text'

with open('08-building-search-applications\\embedding_index_3m.json') as f:
    videos = json.load(f)
    for video in videos:
        video['nomic'] = ollama.embeddings(model=model, prompt=video['summary'])['embedding']
        video.pop('ada_v2')
    newJson = json.dumps(videos)

with open('08-building-search-applications\\embedding_index_nomic_3m.json', 'w') as file:
    # write
    file.write(newJson)