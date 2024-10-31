import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import os


EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "videos"
CHROMA_DATA_PATH = "/Users/dhruvmehrottra007/Desktop/DejaVu/chroma_db"

if not os.path.exists(CHROMA_DATA_PATH):
    os.makedirs(CHROMA_DATA_PATH)

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH, settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name = EMBED_MODEL)

try:
    collection = client.get_collection(name=COLLECTION_NAME)
except chromadb.errors.CollectionNotFoundError:
    collection = chromadb.create_collection(ame=COLLECTION_NAME, embedding_function=embedding_func, metadata={"hnsw:space": "cosine"})

def add(description, video_path, start_time):
    embedding = embedding_func([description])
    id = f"{video_path}_{start_time}"
    collection.add(documents=[description],
                   ids=[id],
                   embeddings=embedding,
                   metadatas=[{"video_path": video_path, "start_time": start_time}])

def query(query):
    embeddings = embedding_func([query])
    results = collection.query(
        query_embeddings=embeddings,
        n_results=3
    )
    video_paths_similar = {}
    for result in results["metadatas"]:
        for result_dict in result:
                print(result_dict["video_path"])
                video_paths_similar[result_dict["start_time"]] = result_dict["video_path"]

    return video_paths_similar


print(query("Ranveer talking about suits"))