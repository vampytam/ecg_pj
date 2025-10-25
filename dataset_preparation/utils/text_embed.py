"""
Text Embedding Utility

This module provides functionality to generate embeddings for text using
the MedEmbed-base-v0.1 model, which is specifically designed for medical text.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Union, List, Optional

class TextEmbedder:
    def __init__(self, model_name = "abhinand/MedEmbed-base-v0.1"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def embed_text(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        embedding = self.model.encode([text])
        return embedding[0]
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return np.array([])
        
        embeddings = self.model.encode(valid_texts)
        return embeddings
    
    def compute_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return np.array([])
        
        similarities = self.model.similarity(embeddings, embeddings)
        return similarities
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         candidate_embeddings: np.ndarray, 
                         top_k: int = 5) -> tuple:
        if candidate_embeddings.size == 0:
            return np.array([]), np.array([])
        
        similarities = self.model.similarity([query_embedding], candidate_embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        return top_indices, top_similarities
    
    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


# Convenience functions for direct usage
def create_embedder(model_name= "abhinand/MedEmbed-base-v0.1") -> TextEmbedder:
    return TextEmbedder(model_name)


def embed_single_text(text, model_name= "abhinand/MedEmbed-base-v0.1") -> np.ndarray:
    embedder = TextEmbedder(model_name)
    return embedder.embed_text(text)


def embed_multiple_texts(texts, model_name = "abhinand/MedEmbed-base-v0.1") -> np.ndarray:
    embedder = TextEmbedder(model_name)
    return embedder.embed_texts(texts)


if __name__ == "__main__":
    import json
    with open ("config.json", "r") as f:
        config = json.load(f)
    
    embedder = TextEmbedder(config["text_embed"]["model_name"])
    
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
        "Patient presents with chest pain and shortness of breath.",
        "ECG shows ST elevation in leads II, III, and aVF."
    ]
    
    embeddings = embedder.embed_texts(sentences)
    print(f"Embeddings shape: {embeddings.shape}")
    
    similarities = embedder.compute_similarity(embeddings)
    print(f"Similarities shape: {similarities.shape}")
    
    query_embedding = embeddings[0]
    top_indices, top_similarities = embedder.find_most_similar(
        query_embedding, embeddings[1:], top_k=3
    )
    
    print(f"Most similar sentences to '{sentences[0]}':")
    for i, (idx, sim) in enumerate(zip(top_indices, top_similarities)):
        print(f"{i+1}. '{sentences[idx+1]}' (similarity: {sim:.4f})")
