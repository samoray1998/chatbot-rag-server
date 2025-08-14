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

  async embedDocuments(texts: string[]): Promise<number[][]> {
    const embeddings: number[][] = [];
    
    for (const text of texts) {
      try {
        // For proper embedding models, you would make a direct API call to Ollama
        // Here's the correct way to get embeddings from Ollama:
        const embedding = await this.getOllamaEmbedding(text);
        embeddings.push(embedding);
      } catch (error) {
        console.error(`Failed to generate embedding for text: ${text.substring(0, 50)}...`, error);
        // Fallback to a simple hash-based embedding
        embeddings.push(this.textToEmbedding(text));
      }
    }
    
    return embeddings;
  }

  async embedQuery(text: string): Promise<number[]> {
    try {
      return await this.getOllamaEmbedding(text);
    } catch (error) {
      console.error(`Failed to generate embedding for query: ${text.substring(0, 50)}...`, error);
      return this.textToEmbedding(text);
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
        throw new Error(`Ollama API error: ${response.status}`);
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

  // Simple hash-based embedding as fallback (not ideal for production)
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
      this.vectorStore = new QdrantVectorStore(this.embeddings, {
        url: this.config.url,
        collectionName: this.config.collectionName,
        apiKey: this.config.apiKey
      });

      // Verify collection exists or create it
      await this.ensureCollection();
      this.isInitialized = true;
      console.log("Vector store initialized successfully");
    } catch (error) {
      console.error("Failed to initialize vector store:", error);
      this.isInitialized = false;
      // Don't throw here - allow graceful degradation
    }
  }

  private async ensureCollection(): Promise<void> {
    if (!this.vectorStore) {
      throw new Error("Vector store not initialized");
    }

    try {
      // Try to create collection if it doesn't exist
      if (this.vectorStore.ensureCollection) {
        await this.vectorStore.ensureCollection();
      }
    } catch (error) {
      console.warn("Could not ensure collection exists:", error);
      // Continue anyway - collection might already exist
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
      console.log(`Added ${documents.length} documents to vector store`);
    } catch (error) {
      console.error("Error adding documents:", error);
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
      return results;
    } catch (error) {
      console.error("Similarity search failed:", error);
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
      const results = await this.vectorStore!.similaritySearchWithScore(query, k, filter);
      return results;
    } catch (error) {
      console.error("Similarity search with score failed:", error);
      return [];
    }
  }

  public async deleteDocuments(filter: object): Promise<void> {
    await this.ensureInitialized();

    try {
      await this.vectorStore!.delete({ filter });
      console.log("Documents deleted successfully");
    } catch (error) {
      console.error("Failed to delete documents:", error);
      throw new Error(`Deletion failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  public async getDocumentCount(): Promise<number> {
    try {
      await this.ensureInitialized();
      // This might need adjustment based on your Qdrant setup
      // As a fallback, we'll return -1 to indicate unknown
      return -1;
    } catch (error) {
      console.error("Failed to get document count:", error);
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

      // Try a simple search to verify connectivity
      await this.vectorStore.similaritySearch("health check", 1);
      
      return { 
        initialized: true, 
        connected: true 
      };
    } catch (error) {
      return { 
        initialized: this.isInitialized, 
        connected: false, 
        error: error instanceof Error ? error.message : String(error)
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