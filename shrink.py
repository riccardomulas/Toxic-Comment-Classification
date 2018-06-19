import pandas as pd
import numpy as np
import os

def shrink_to_vocabulary(embeddings_input_path, vocabulary):
    embeddings = load(embeddings_input_path, vocabulary)
    dirname, filename = os.path.split(embeddings_input_path)
    outputdir = os.path.join(dirname, 'shrunk')
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    embeddings_output_path = os.path.join(outputdir, filename)
    save(embeddings, embeddings_output_path)


def save(embeddings, path):
    with open(path, 'a') as f:
        for word in embeddings.keys():
            vector = embeddings[word]
            values = ' '.join(map(str, vector))
            f.write(word + ' ' + values + '\n')

shrink_to_vocabulary(embeddings_input_path, vocabulary)
