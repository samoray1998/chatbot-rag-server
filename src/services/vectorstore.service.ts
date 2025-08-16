import { QdrantVectorStore } from "@langchain/qdrant";
import { Ollama } from "@langchain/ollama";
import { Document } from "@langchain/core/documents";
import { Embeddings } from "@langchain/core/embeddings";

interface VectorStoreConfig {
  url: string;
  collectionName: string;
  apiKey?: string;
}

// Create a custom Ollama Embeddings class for proper embedding models
class OllamaEmbeddings extends Embeddings {
  private ollama: Ollama;
  private embeddingDimensions: number | null = null;

  constructor(config: {
    baseUrl?: string;
    model: string;
    temperature?: number;
  }) {
    super({});
    this.ollama = new Ollama({
      baseUrl: config.baseUrl || "http://ollama:11434",
      model: config.model,
      temperature: config.temperature || 0.0, // Lower temperature for embeddings
    });
  }

  // Get the actual dimensions from the model
  async getEmbeddingDimensions(): Promise<number> {
    if (this.embeddingDimensions !== null) {
      return this.embeddingDimensions;
    }

    try {
      // Test with a simple query to get actual dimensions
      const testEmbedding = await this.getOllamaEmbedding("test");
      this.embeddingDimensions = testEmbedding.length;
      console.log(`✅ Detected embedding dimensions: ${this.embeddingDimensions}`);
      return this.embeddingDimensions;
    } catch (error) {
      console.error("Failed to detect embedding dimensions:", error);
      // Default fallback - but this might cause issues
      this.embeddingDimensions = 384;
      return this.embeddingDimensions;
    }
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    const embeddings: number[][] = [];
    
    // Ensure we know the dimensions
    await this.getEmbeddingDimensions();
    
    for (const text of texts) {
      try {
        // For proper embedding models, you would make a direct API call to Ollama
        const embedding = await this.getOllamaEmbedding(text);
        embeddings.push(embedding);
      } catch (error) {
        console.error(`Failed to generate embedding for text: ${text.substring(0, 50)}...`, error);
        // Fallback to a simple hash-based embedding with correct dimensions
        embeddings.push(this.textToEmbedding(text, this.embeddingDimensions || 384));
      }
    }
    
    return embeddings;
  }

  async embedQuery(text: string): Promise<number[]> {
    try {
      const embedding = await this.getOllamaEmbedding(text);
      console.log(`Query embedding generated with ${embedding.length} dimensions`);
      return embedding;
    } catch (error) {
      console.error(`Failed to generate embedding for query: ${text.substring(0, 50)}...`, error);
      // Ensure we use the correct dimensions for fallback
      await this.getEmbeddingDimensions();
      return this.textToEmbedding(text, this.embeddingDimensions || 384);
    }
  }

  // Direct API call to Ollama for embeddings (requires embedding model)
  private async getOllamaEmbedding(text: string): Promise<number[]> {
    try {
      const baseUrl = (this.ollama as any).baseUrl || "http://ollama:11434";
      const model = (this.ollama as any).model;
      
      // Make direct API call to Ollama embeddings endpoint
      const response = await fetch(`${baseUrl}/api/embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: model,
          prompt: text,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Ollama API error: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      
      if (data.embedding && Array.isArray(data.embedding)) {
        return data.embedding;
      } else {
        throw new Error('Invalid embedding response from Ollama');
      }
    } catch (error) {
      console.error('Error calling Ollama embeddings API:', error);
      throw error;
    }
  }

  // Simple hash-based embedding as fallback with configurable dimensions
  private textToEmbedding(text: string, dimensions: number = 384): number[] {
    const embedding = new Array(dimensions).fill(0);
    
    for (let i = 0; i < text.length; i++) {
      const charCode = text.charCodeAt(i);
      embedding[i % dimensions] += charCode;
    }
    
    // Normalize the embedding
    const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return embedding.map(val => magnitude > 0 ? val / magnitude : 0);
  }
}

class VectorStoreService {
  private vectorStore: QdrantVectorStore | null = null;
  private embeddings: OllamaEmbeddings;
  private isInitialized = false;
  private config: VectorStoreConfig;

  constructor(config: VectorStoreConfig) {
    this.config = config;
    
    // Initialize Ollama embeddings
    this.embeddings = new OllamaEmbeddings({
      baseUrl: process.env.OLLAMA_URL || "http://ollama:11434",
      model: "dengcao/Qwen3-Embedding-0.6B:Q8_0", // Your embedding model
      temperature: 0.0, // Use 0 temperature for consistent embeddings
    });

    // Initialize asynchronously
    this.initializeVectorStore();
  }

  private async initializeVectorStore(): Promise<void> {
    try {
      console.log("🔄 Initializing vector store...");
      
      // Get actual embedding dimensions from the model
      const embeddingDims = await this.embeddings.getEmbeddingDimensions();
      console.log(`📏 Using embedding dimensions: ${embeddingDims}`);

      this.vectorStore = new QdrantVectorStore(this.embeddings, {
        url: this.config.url,
        collectionName: this.config.collectionName,
        apiKey: this.config.apiKey
      });

      // Verify collection exists and check dimensions
      await this.ensureCollection(embeddingDims);
      this.isInitialized = true;
      console.log("✅ Vector store initialized successfully");
    } catch (error) {
      console.error("❌ Failed to initialize vector store:", error);
      this.isInitialized = false;
      // Don't throw here - allow graceful degradation
    }
  }

  private async ensureCollection(expectedDimensions: number): Promise<void> {
    if (!this.vectorStore) {
      throw new Error("Vector store not initialized");
    }

    try {
      // Get collection info to check if dimensions match
      const qdrantUrl = this.config.url;
      const collectionName = this.config.collectionName;
      
      const response = await fetch(`${qdrantUrl}/collections/${collectionName}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          ...(this.config.apiKey && { 'api-key': this.config.apiKey })
        }
      });

      if (response.ok) {
        const collectionInfo = await response.json();
        const actualDimensions = collectionInfo.result?.config?.params?.vectors?.size;
        
        console.log(`📊 Collection exists with ${actualDimensions} dimensions, expected ${expectedDimensions}`);
        
        if (actualDimensions && actualDimensions !== expectedDimensions) {
          console.error(`❌ DIMENSION MISMATCH!`);
          console.error(`   Collection: ${actualDimensions} dimensions`);
          console.error(`   Model: ${expectedDimensions} dimensions`);
          console.error(`   
Solutions:
1. Delete and recreate collection: DELETE ${qdrantUrl}/collections/${collectionName}
2. Or change embedding model to one that produces ${actualDimensions} dimensions
3. Or manually recreate collection with ${expectedDimensions} dimensions`);
          
          throw new Error(`Dimension mismatch: collection=${actualDimensions}, model=${expectedDimensions}`);
        }
      } else if (response.status === 404) {
        // Collection doesn't exist, create it with correct dimensions
        console.log(`📝 Creating new collection with ${expectedDimensions} dimensions`);
        
        const createResponse = await fetch(`${qdrantUrl}/collections/${collectionName}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
            ...(this.config.apiKey && { 'api-key': this.config.apiKey })
          },
          body: JSON.stringify({
            vectors: {
              size: expectedDimensions,
              distance: 'Cosine'
            }
          })
        });

        if (!createResponse.ok) {
          throw new Error(`Failed to create collection: ${createResponse.status}`);
        }
        
        console.log(`✅ Created collection with ${expectedDimensions} dimensions`);
      } else {
        throw new Error(`Failed to check collection: ${response.status}`);
      }
      
      // Try to ensure collection through LangChain (fallback)
      if (this.vectorStore.ensureCollection) {
        await this.vectorStore.ensureCollection();
      }
    } catch (error) {
      console.error("❌ Collection setup failed:", error);
      throw error;
    }
  }

  private async ensureInitialized(): Promise<void> {
    if (!this.isInitialized || !this.vectorStore) {
      await this.initializeVectorStore();
      if (!this.isInitialized || !this.vectorStore) {
        throw new Error("Vector store initialization failed");
      }
    }
  }

  public async addDocuments(documents: Document[]): Promise<void> {
    await this.ensureInitialized();

    try {
      await this.vectorStore!.addDocuments(documents);
      console.log(`✅ Added ${documents.length} documents to vector store`);
    } catch (error) {
      console.error("❌ Error adding documents:", error);
      throw new Error(`Failed to add documents: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  public async similaritySearch(
    query: string,
    k = 4,
    filter?: object
  ): Promise<Document[]> {
    try {
      await this.ensureInitialized();
      const results = await this.vectorStore!.similaritySearch(query, k, filter);
      console.log(`🔍 Similarity search returned ${results.length} results`);
      return results;
    } catch (error) {
      console.error("❌ Similarity search failed:", error);
      // Return empty results instead of throwing to allow graceful degradation
      return [];
    }
  }

  public async similaritySearchWithScore(
    query: string,
    k = 4,
    filter?: object
  ): Promise<[Document, number][]> {
    try {
      await this.ensureInitialized();
      console.log(`🔍 Starting similarity search with score for: "${query.substring(0, 50)}..."`);
      
      const results = await this.vectorStore!.similaritySearchWithScore(query, k, filter);
      console.log(`✅ Similarity search returned ${results.length} results with scores`);
      
      // Log scores for debugging
      results.forEach(([doc, score], index) => {
        console.log(`   ${index + 1}. Score: ${score.toFixed(3)} - ${doc.metadata?.source || 'Unknown source'}`);
      });
      
      return results;
    } catch (error) {
      console.error("❌ Similarity search with score failed:", error);
       const isError = error instanceof Error;
      const hasData = error && typeof error === 'object' && 'data' in error;
      const errorData = hasData ? (error as any).data : null;
      
      // Enhanced error logging for dimension mismatches
      if ( (isError && error.message?.includes('dimension')) ||
        (errorData?.status?.error?.includes('dimension'))) {
        console.error("🚨 This is a DIMENSION MISMATCH error!");
        console.error("   Your embedding model and Qdrant collection have different vector dimensions");
        console.error("   Check the logs above for specific dimension numbers and solutions");
      }
      
      return [];
    }
  }

  public async deleteDocuments(filter: object): Promise<void> {
    await this.ensureInitialized();

    try {
      await this.vectorStore!.delete({ filter });
      console.log("✅ Documents deleted successfully");
    } catch (error) {
      console.error("❌ Failed to delete documents:", error);
      throw new Error(`Deletion failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  public async getDocumentCount(): Promise<number> {
    try {
      await this.ensureInitialized();
      // This might need adjustment based on your Qdrant setup
      const response = await fetch(`${this.config.url}/collections/${this.config.collectionName}`, {
        headers: {
          ...(this.config.apiKey && { 'api-key': this.config.apiKey })
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        return data.result?.points_count || 0;
      }
      
      return -1;
    } catch (error) {
      console.error("❌ Failed to get document count:", error);
      return -1;
    }
  }

  public isReady(): boolean {
    return this.isInitialized && !!this.vectorStore;
  }

  public getConfig(): VectorStoreConfig {
    return { ...this.config };
  }

  public async healthCheck(): Promise<{ 
    initialized: boolean; 
    connected: boolean; 
    embeddingDimensions?: number;
    error?: string 
  }> {
    try {
      if (!this.isInitialized || !this.vectorStore) {
        return { 
          initialized: false, 
          connected: false, 
          error: "Vector store not initialized" 
        };
      }

      // Get embedding dimensions
      const dimensions = await this.embeddings.getEmbeddingDimensions();

      // Try a simple search to verify connectivity
      await this.vectorStore.similaritySearch("health check", 1);
      
      return { 
        initialized: true, 
        connected: true,
        embeddingDimensions: dimensions
      };
    } catch (error) {
      return { 
        initialized: this.isInitialized, 
        connected: false, 
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  // Utility method to check and fix dimension mismatches
  public async diagnoseDimensions(): Promise<{
    modelDimensions: number;
    collectionDimensions: number | null;
    match: boolean;
    suggestions: string[];
  }> {
    try {
      const modelDims = await this.embeddings.getEmbeddingDimensions();
      
      // Check collection dimensions
      const response = await fetch(`${this.config.url}/collections/${this.config.collectionName}`, {
        headers: {
          ...(this.config.apiKey && { 'api-key': this.config.apiKey })
        }
      });
      
      let collectionDims = null;
      if (response.ok) {
        const data = await response.json();
        collectionDims = data.result?.config?.params?.vectors?.size || null;
      }
      
      const match = collectionDims === modelDims;
      const suggestions: string[] = [];
      
      if (!match && collectionDims) {
        suggestions.push(`Current: Model=${modelDims}d, Collection=${collectionDims}d`);
        suggestions.push(`Option 1: Delete collection and recreate with ${modelDims} dimensions`);
        suggestions.push(`Option 2: Switch to embedding model that produces ${collectionDims} dimensions`);
        suggestions.push(`Command to delete: curl -X DELETE "${this.config.url}/collections/${this.config.collectionName}"`);
      }
      
      return {
        modelDimensions: modelDims,
        collectionDimensions: collectionDims,
        match,
        suggestions
      };
    } catch (error) {
      return {
        modelDimensions: 0,
        collectionDimensions: null,
        match: false,
        suggestions: [`Error diagnosing: ${error}`]
      };
    }
  }
}

// Initialize with your Qdrant configuration
export const vectorStoreService = new VectorStoreService({
  url: process.env.QDRANT_URL || "http://qdrant:6333",
  collectionName: process.env.QDRANT_COLLECTION || "chatbot_docs",
  apiKey: process.env.QDRANT_API_KEY
});

export default VectorStoreService;