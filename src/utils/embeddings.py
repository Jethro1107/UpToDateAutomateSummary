import chromadb
from typing import List
from ollama import embed

def store_embeddings(texts: List[str], sources: List[str]) -> chromadb.Collection:
    """
    Store text embeddings in a ChromaDB collection.

    Args:
        texts (List[str]): List of text strings to embed.
        sources (List[str]): List of source filenames corresponding to texts.

    Returns:
        chromadb.Collection: ChromaDB collection with stored embeddings.
    """
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="notes")
    if not texts:
        print("Warning: No texts to embed, returning empty collection")
        return collection
    for i, (text, source) in enumerate(zip(texts, sources)):
        if not isinstance(text, str):
            print(f"Error: Invalid text type {type(text)} for source {source}, skipping")
            continue
        try:
            response = embed(model="mxbai-embed-large", input=text)
            embeddings = response["embeddings"]
            if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
                embedding_vector = embeddings[0]
            else:
                embedding_vector = embeddings
            if not all(isinstance(x, (int, float)) for x in embedding_vector):
                print(f"Error: Invalid embedding format for source {source}, expected list of floats, got {type(embedding_vector[0])}")
                continue
            collection.add(
                embeddings=[embedding_vector],
                documents=[text],
                metadatas=[{"source": source}],
                ids=[f"doc_{i}"]
            )
            print(f"Successfully added embedding for source {source}")
        except Exception as e:
            print(f"Error processing embedding for source {source}: {str(e)}")
            continue
    return collection