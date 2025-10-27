import numpy as np
import chromadb
import json
import xxhash
import uuid

from .text_embed import TextEmbedder

class VectorDBClient:
    _instance = None
    _config = None
    _collection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorDBClient, cls).__new__(cls)
            cls._config = cls._load_config()
            cls._collection = cls._setup_chroma_client_and_collection()
        return cls._instance

    @classmethod
    def _load_config(cls):
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)

    @classmethod
    def _setup_chroma_client_and_collection(cls):
        chroma_client = chromadb.PersistentClient(cls._config["vectordb"]["db_path"])
        collection_name = f"{cls._config['text_embed']['model_name'].replace('/', '_')}_{cls._config['vectordb']['db_name']}"
        return chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def emb_to_uuid_hash(v: np.ndarray) -> str:
        return xxhash.xxh64(v.astype('<f4').tobytes()).hexdigest()

    def add_doc_to_db(self, doc, embedding):
        self._collection.add(
            ids=[str(self.emb_to_uuid_hash(embedding))],
            embeddings=[embedding],
            documents=[doc]
        )

    def search_most_similars_by_embeddings(self, query_embs: np.ndarray, dist_threshold: float = 0.5):
        results = self._collection.query(
            query_embeddings=query_embs.tolist(),
            n_results=3
        )

        docs_ = results.get("documents", [])
        distances_ = results.get("distances", [])

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
    client = VectorDBClient()
    litfl_fp = './dataset_preparation/litfl_crawler/new_litfl.json'
    with open(litfl_fp, 'r', encoding='utf-8') as f:
        diagnosis_infos = json.load(f)
    embedder = TextEmbedder(client._config["text_embed"]["model_name"])
    for title, doc in diagnosis_infos.items():
        embedding = embedder.embed_text(title)
        client.add_doc_to_db(doc, embedding)

    query_texts = ["Premature complexes"]
    query_embs = np.array([embedder.embed_text(text) for text in query_texts])
    docs, dists = client.search_most_similars_by_embeddings(query_embs, dist_threshold=0.2)

    print(docs)
    print(dists)
    

