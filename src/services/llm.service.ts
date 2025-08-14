import { Ollama } from "@langchain/ollama";
import { getCacheService } from "./cache.service";
import { vectorStoreService } from "./vectorstore.service";
import type CacheService from "./cache.service";
import type { BaseMessage } from "@langchain/core/messages";
import { Document } from "@langchain/core/documents";

class LLMService {
  private ollama: Ollama;
  private cacheService: CacheService | null = null;
  private isInitialized = false;

  constructor() {
    this.ollama = new Ollama({
      baseUrl: process.env.OLLAMA_URL || "http://ollama:11434",
      model: "mistral:7b-instruct-v0.2-q3_K_S",
      temperature: 0.7,
      // Note: We'll handle caching manually instead of using LangChain's built-in cache
      // to have more control over the caching logic
    });

    // Initialize cache service asynchronously
    this.initializeCache();
  }

  private async initializeCache(): Promise<void> {
    try {
      console.log("🔄 Initializing cache service...");
      this.cacheService = await getCacheService();

      // Test the cache service
      const testStatus = this.cacheService.getStatus();
      console.log("📊 Cache service status:", testStatus);

      if (!testStatus.connected) {
        throw new Error("Cache service not connected");
      }

      this.isInitialized = true;
      console.log("✅ LLM Service cache initialized successfully");
    } catch (error) {
      console.error("❌ Failed to initialize cache service:", error);
      this.cacheService = null; // ❗ Important: Set to null on failure
      this.isInitialized = true; // Still set to true to allow service to work without cache
    }
  }

  private async ensureInitialized(): Promise<void> {
    if (!this.isInitialized) {
      await this.initializeCache();
    }
  }

  async generate(prompt: string): Promise<string> {
    try {
      console.log("🔄 Starting generate() method");
      console.log("📊 Initial state - isInitialized:", this.isInitialized);
      console.log("📊 Initial state - cacheService exists:", !!this.cacheService);
      await this.ensureInitialized();

      console.log("📊 After ensureInitialized - isInitialized:", this.isInitialized);
      console.log("📊 After ensureInitialized - cacheService exists:", !!this.cacheService);
      console.log("📊 Cache service ready:", this.cacheService?.isReady());

      // Create a cache key for this prompt
      const cacheKey = this.createCacheKey(prompt);
      console.log("🔑 Generated cache key:", cacheKey);

      // Check cache first
      if (this.cacheService) {
        console.log("🔍 Attempting cache lookup...");
        const cached = await this.cacheService.lookup(cacheKey);
        console.log("📋 Cache lookup result:", cached ? "HIT" : "MISS");
        if (cached) {
          console.log("✅ Cache hit for prompt");
          return cached;
        }
      } else {
        console.log("❌ No cache service available for lookup");
      }

      console.log("Cache miss, generating new response");

      // Generate via Ollama
      const response = await this.ollama.invoke(prompt);

      // Extract the response content - handle different response types
      const responseText = this.extractResponseText(response);
      console.log("📝 Generated response length:", responseText.length);

      // Store in cache if available
      if (this.cacheService) {
        console.log("💾 Attempting to cache response...");
        try {
          await this.cacheService.update(cacheKey, responseText);
          console.log("✅ Response cached successfully");
        } catch (cacheError) {
          console.error("❌ Cache update failed:", cacheError);
        }
      } else {
        console.log("❌ No cache service available for storage");
      }

      return responseText;
    } catch (error) {
      console.error("💥 Generate method error:", error);
      console.error("Ollama error:", error);
      return `Error generating response: ${error instanceof Error ? error.message : String(error)}`;
    }
  }

  async generateWithContext(prompt: string, options?: {
    maxDocs?: number;
    minScore?: number;
    includeScores?: boolean;
  }): Promise<string> {
    try {
      await this.ensureInitialized();

      const { maxDocs = 3, minScore = 0.0, includeScores = false } = options || {};

      // Create cache key that includes context for RAG queries
      const contextCacheKey = this.createCacheKey(`rag:${prompt}:docs${maxDocs}:score${minScore}`);

      // Check cache first for RAG responses
      if (this.cacheService) {
        const cached = await this.cacheService.lookup(contextCacheKey);
        if (cached) {
          console.log("Cache hit for RAG query");
          return cached;
        }
      }

      // 1. Check if vector store is available
      if (!vectorStoreService.isReady()) {
        console.warn("Vector store not ready, falling back to basic generation");
        return await this.generate(prompt);
      }

      // 2. Retrieve relevant documents with scores if requested
      let relevantDocs: any[] = [];
      if (includeScores) {
        const docsWithScores = await vectorStoreService.similaritySearchWithScore(prompt, maxDocs);
        relevantDocs = docsWithScores
          .filter(([doc, score]) => score >= minScore)
          .map(([doc, score]) => ({ ...doc, score }));
      } else {
        const docs = await vectorStoreService.similaritySearch(prompt, maxDocs);
        relevantDocs = docs;
      }

      // 3. Handle case where no relevant documents are found
      if (relevantDocs.length === 0) {
        console.log("No relevant documents found, using basic generation");
        const response = await this.generate(`Answer this question: ${prompt}\n\nNote: No specific context documents were available.`);

        // Cache this response too
        if (this.cacheService) {
          await this.cacheService.update(contextCacheKey, response);
        }

        return response;
      }

      // 4. Format context with metadata
      const context = relevantDocs
        .map((doc, index) => {
          const source = doc.metadata?.source || `Document ${index + 1}`;
          const score = doc.score ? ` (relevance: ${doc.score.toFixed(3)})` : '';
          return `SOURCE: ${source}${score}\nCONTENT: ${doc.pageContent}`;
        })
        .join("\n\n---\n\n");

      // 5. Create enhanced prompt
      const augmentedPrompt = `You are a helpful assistant that answers questions based on the provided context.

    INSTRUCTIONS:
    - Use the context below to answer the question
    - If the context doesn't contain enough information to fully answer the question, say so clearly
    - Be specific and cite relevant parts of the context when possible
    - If multiple sources provide conflicting information, acknowledge this

    CONTEXT:
    ${context}

    QUESTION: ${prompt}

    Please provide a comprehensive answer based on the context above:`;

      // 6. Generate response
      const response = await this.ollama.invoke(augmentedPrompt);
      const responseText = this.extractResponseText(response);

      // 7. Cache the RAG response
      if (this.cacheService) {
        await this.cacheService.update(contextCacheKey, responseText);
        console.log("RAG response cached successfully");
      }

      return responseText;
    } catch (error) {
      console.error("Error in generateWithContext:", error);
      // Fallback to basic generation
      console.log("Falling back to basic generation due to error");
      return await this.generate(prompt);
    }
  }

  /**
   * Add documents to the vector store
   */
  async addDocuments(documents: { content: string; metadata?: Record<string, any> }[]): Promise<void> {
    try {
      const docs = documents.map(doc => new Document({
        pageContent: doc.content,
        metadata: doc.metadata || {}
      }));

      await vectorStoreService.addDocuments(docs);
      console.log(`Successfully added ${documents.length} documents to vector store`);
    } catch (error) {
      console.error("Failed to add documents to vector store:", error);
      throw error;
    }
  }

  /**
   * Search for similar documents without generating a response
   */
  async searchDocuments(
    query: string,
    options?: {
      maxResults?: number;
      includeScores?: boolean;
      filter?: object;
    }
  ): Promise<any[]> {
    const { maxResults = 5, includeScores = false, filter } = options || {};

    try {
      if (includeScores) {
        return await vectorStoreService.similaritySearchWithScore(query, maxResults, filter);
      } else {
        return await vectorStoreService.similaritySearch(query, maxResults, filter);
      }
    } catch (error) {
      console.error("Document search failed:", error);
      return [];
    }
  }

  /**
   * Extract text response from Ollama response object
   */
  private extractResponseText(response: any): string {
    // Handle different response types from Ollama
    if (typeof response === 'string') {
      return response;
    }

    // Handle AIMessage or similar objects
    if (response && typeof response === 'object') {
      // Try different possible properties
      if (response.content) return String(response.content);
      if (response.text) return String(response.text);
      if (response.message) return String(response.message);
      if (response.output) return String(response.output);
      if (response.response) return String(response.response);

      // If it has a toString method
      if (response.toString && typeof response.toString === 'function') {
        const str = response.toString();
        if (str !== '[object Object]') return str;
      }
    }

    // Fallback to string conversion
    return String(response);
  }

  /**
   * Generate a streaming response (if Ollama supports it)
   */
  async generateStream(prompt: string, onToken?: (token: string) => void): Promise<string> {
    try {
      await this.ensureInitialized();

      const cacheKey = this.createCacheKey(prompt);

      // Check cache first
      if (this.cacheService) {
        const cached = await this.cacheService.lookup(cacheKey);
        if (cached) {
          // Simulate streaming for cached responses
          if (onToken) {
            const tokens = cached.split(' ');
            for (const token of tokens) {
              onToken(token + ' ');
              await new Promise(resolve => setTimeout(resolve, 50)); // Simulate delay
            }
          }
          return cached;
        }
      }

      // For streaming, we'll need to collect the full response to cache it
      let fullResponse = '';

      // Note: This depends on whether your Ollama setup supports streaming
      // You might need to adjust this based on your LangChain Ollama version
      const response = await this.ollama.invoke(prompt);
      fullResponse = this.extractResponseText(response);

      // Cache the response
      if (this.cacheService) {
        await this.cacheService.update(cacheKey, fullResponse);
      }

      // Simulate streaming output
      if (onToken) {
        const tokens = fullResponse.split(' ');
        for (const token of tokens) {
          onToken(token + ' ');
          await new Promise(resolve => setTimeout(resolve, 50));
        }
      }

      return fullResponse;
    } catch (error) {
      console.error("Streaming generation error:", error);
      throw error;
    }
  }

  /**
   * Clear cache for a specific prompt or pattern
   */
  async clearCache(pattern?: string): Promise<void> {
    if (this.cacheService) {
      await this.cacheService.flushCache(pattern || "llm:*");
      console.log("Cache cleared successfully");
    }
  }

  /**
   * Get cache statistics
   */
  async getCacheStats(): Promise<{ connected: boolean; client: string }> {
    if (!this.cacheService) {
      return { connected: false, client: "Cache service not initialized" };
    }
    return this.cacheService.getStatus();
  }

  /**
   * Create a consistent cache key for prompts
   */
  private createCacheKey(prompt: string): string {
    const modelKey = `${this.ollama.model || 'unknown-model'}`;
    const temperature = this.ollama.temperature || 0.7;
    return `llm:${modelKey}:temp${temperature}:${this.hashPrompt(prompt)}`;
  }

  /**
   * Simple hash function for cache keys
   */
  private hashPrompt(prompt: string): string {
    let hash = 0;
    for (let i = 0; i < prompt.length; i++) {
      const char = prompt.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36);
  }

  /**
   * Health check for the service
   */
  async healthCheck(): Promise<{
    ollama: boolean;
    cache: boolean;
    vectorStore: boolean;
    details: any;
  }> {
    const health = {
      ollama: false,
      cache: false,
      vectorStore: false,
      details: {} as any
    };

    // Test Ollama
    try {
      await this.ollama.invoke("test");
      health.ollama = true;
    } catch (error) {
      health.details.ollama = error instanceof Error ? error.message : String(error);
    }

    // Test Cache
    try {
      await this.ensureInitialized();
      if (this.cacheService) {
        const status = this.cacheService.getStatus();
        health.cache = status.connected;
        health.details.cache = status;
      }
    } catch (error) {
      health.details.cache = error instanceof Error ? error.message : String(error);
    }

    // Test Vector Store
    try {
      const vectorHealth = await vectorStoreService.healthCheck();
      health.vectorStore = vectorHealth.connected;
      health.details.vectorStore = vectorHealth;
    } catch (error) {
      health.details.vectorStore = error instanceof Error ? error.message : String(error);
    }

    return health;
  }

  /**
   * Graceful shutdown
   */
  async shutdown(): Promise<void> {
    if (this.cacheService) {
      await this.cacheService.disconnect();
    }
  }
}

// Create singleton instance
export const llmService = new LLMService();

// Handle graceful shutdown
process.on('SIGTERM', async () => {
  await llmService.shutdown();
});

process.on('SIGINT', async () => {
  await llmService.shutdown();
});

export default LLMService;