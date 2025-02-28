import pinecone

class GaitDatabase:
    def __init__(self, api_key, environment, index_name, dimension=512):
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )
            
        self.index = pinecone.Index(index_name)
        
    def store_gait_embedding(self, id, embedding, metadata=None):
        """Store a gait embedding in Pinecone."""
        self.index.upsert(vectors=[(id, embedding.tolist(), metadata)])
        
    def search_similar_gaits(self, query_embedding, top_k=5):
        """Find similar gait patterns."""
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        return results
