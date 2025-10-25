import numpy as np
import chromadb
import json
import hashlib
import uuid


from .text_embed import TextEmbedder

def load_config():
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

config = load_config()

# setup chroma client and collection
def setup_chroma_client_and_collection(chroma_client_path, collection_name):
    chroma_client = chromadb.PersistentClient(chroma_client_path)
    collection = chroma_client.get_or_create_collection(name=collection_name,
                                                        embedding_function=None,
                                                        metadata={"hnsw:space": "cosine"}) # distance metric is (1-consine_similarity)

    return collection

collection = setup_chroma_client_and_collection(config["vectordb"]["db_path"], 
                                                f"{config["text_embed"]["model_name"].replace("/", "_")}_{config["vectordb"]["db_name"]}")

def emb_to_uuid_hash(v: np.ndarray) -> uuid.UUID:
    h = hashlib.md5(v.astype('<f4').tobytes()).digest()
    return uuid.UUID(bytes=h)

def add_doc_to_db(doc, embedding):
    collection.add(
        ids = [str(emb_to_uuid_hash(embedding))],
        embeddings=[embedding],
        documents=[doc]
    )

def search_most_similars_by_embeddings(query_embs: np.ndarray, dist_threshold: float = 0.5):
    """
    results is of Class:
        class QueryResult(TypedDict):
            ids: List[IDs]
            embeddings: Optional[List[Embeddings]],
            documents: Optional[List[List[Document]]]
            metadatas: Optional[List[List[Metadata]]]
            distances: Optional[List[List[float]]]
            included: Include
    """
    results = collection.query(
        query_embeddings=query_embs.tolist(),
        n_results=1
    )

    docs_ = results.get("documents", [])
    # distance metric is (1-consine_similarity)
    distances_ = results.get("distances", [])
    print(docs_)
    print(distances_)
    
    filtered_docs = []
    filtered_dists = []
    for doc_list, dist_list in zip(docs_, distances_):
        new_docs = []
        new_dists = []
        for doc, dist in zip(doc_list, dist_list):
            if dist <= dist_threshold:
                new_docs.append(doc)
                new_dists.append(dist)
        filtered_docs.append(new_docs)
        filtered_dists.append(new_dists)

    return filtered_docs, filtered_dists


if __name__ == "__main__":
    documents=[
        "Chest pain",
        "Dizziness, lightheadedness or confusion",
        "Pounding, skipping or fluttering heartbeat",
        "Fast pulse",
        "Shortness of breath",
        "Weakness or fatigue    ",
        "Reduced ability to exercise"
    ]
    embedder = TextEmbedder(config["text_embed"]["model_name"])
    for doc in documents:
        embedding = embedder.embed_text(doc)
        add_doc_to_db(doc, embedding)
    
    query_texts = ["respiratory difficulty"]
    query_embs = np.array([embedder.embed_text(text) for text in query_texts])
    docs, dists = search_most_similars_by_embeddings(query_embs, dist_threshold=0.5)
    
    print(docs)
    print(dists)
    

