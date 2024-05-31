import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from textGen import TextTokenizer

class ConfidenceScorer(nn.Module):
    def __init__(self, model_file):
        # Load the model from the file
        self.model = TextTokenizer(model_file)

    def score(self, user_prompt):
        # Convert the user prompt to a vector
        query_vector = self.prompt_to_vector(user_prompt)

        # Get the weights of the model's first layer
        model_weights = self.model.layers[0].get_weights()[0]

        # Reshape the vectors to 2D arrays
        model_weights_2d = model_weights.reshape(1, -1)
        query_vector_2d = query_vector.reshape(1, -1)

        # Calculate the cosine similarity
        similarity = cosine_similarity(model_weights_2d, query_vector_2d)

        # Return the similarity score
        return similarity[0][0]

def score(self, user_prompt):
    # Convert the user prompt to a vector
    query_vector = self.prompt_to_vector(user_prompt)

    # Get the weights of the model's first layer
    model_weights = self.model.layers[0].get_weights()[0]

    # Reshape the vectors to 2D arrays
    model_weights_2d = model_weights.reshape(1, -1)
    query_vector_2d = query_vector.reshape(1, -1)

    # Calculate the cosine similarity
    similarity = cosine_similarity(model_weights_2d, query_vector_2d)[0][0]

    # Categorize the confidence score based on the cosine similarity
    if similarity >= 0.6:
        confidence_score = 1  # Confident
    else:
        confidence_score = 0  # No confidence

    return confidence_score