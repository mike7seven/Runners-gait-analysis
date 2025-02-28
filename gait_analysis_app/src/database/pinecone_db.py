from pinecone import Pinecone, ServerlessSpec

class GaitDatabase:
    def __init__(self, api_key, environment, index_name, dimension):
        """Initialize connection to Pinecone"""
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension

        self.pc = Pinecone(api_key=self.api_key)
        
        # Create index if it doesn't exist
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region=environment.split('-')[0] 
                )
            )
            
        self.index = self.pc.Index(index_name)
        
    def store_gait_embedding(self, id, embedding, metadata):
        """Store a gait embedding in Pinecone."""
        self.index.upsert(vectors=[(id, embedding, metadata)])
        
    def search_similar_gaits(self, query_embedding, top_k=5):
        """Find similar gait patterns."""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results
