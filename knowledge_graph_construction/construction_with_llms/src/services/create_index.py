from dotenv import load_dotenv
load_dotenv()

import os
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph


embedding_provider = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

embedding_dim = embedding_provider._client.get_sentence_embedding_dimension()

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)


graph.query(
    """
    CREATE VECTOR INDEX `chunkVector`
    IF NOT EXISTS
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {indexConfig: {
    `vector.dimensions`: $embedding_dim,
    `vector.similarity_function`: 'cosine'
    }};
    """,
    params={"embedding_dim": embedding_dim}
)
