import chromadb

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="litfl")

def add_to_db(text, embedding):
    collection.add(
        embeddings=[embedding],
        metadatas=[{"text": text}]
    )

def get_from_db(text, embedding): 
    return collection.get(
        query_embeddings=[embedding],
        n_results=1,
        where={"text": {"$eq": text}}
    )
    
def get_all_from_db():
    return collection.get_all()

def delete_from_db(text):
    collection.delete(
        where={"text": {"$eq": text}}
    )


def delete_all_from_db():
    collection.delete_all()    
    
def get_all_texts_from_db():
    results = collection.get_all()
    texts = [item['text'] for item in results['metadatas']]
    return texts


def get_all_embeddings_from_db():
    results = collection.get_all()
    embeddings = [item['embedding'] for item in results['embeddings']]
    return embeddings


def get_all_texts_and_embeddings_from_db():
    results = collection.get_all()
    texts = [item['text'] for item in results['metadatas']]
    embeddings = [item['embedding'] for item in results['embeddings']]
    return texts, embeddings


if __name__ == "__main__":
    texts, embeddings = get_all_texts_and_embeddings_from_db()
    print(texts)
    print(embeddings)
    print(len(texts))
    print(len(embeddings))
    print(texts[0])
    print(embeddings[0])
    

