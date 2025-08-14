import { createClient, type RedisClientType } from "redis";

// Option 1: Using @langchain/redis (requires: npm install @langchain/redis)
// import { RedisCache } from "@langchain/redis";

// Option 2: Using @langchain/community (alternative import path)
// import { RedisCache } from "@langchain/community/caches/ioredis";

// Option 3: Create a simple cache interface without LangChain dependency
interface CacheInterface {
  lookup(key: string): Promise<string | null>;
  update(key: string, value: string): Promise<void>;
}

class SimpleCacheAdapter implements CacheInterface {
  constructor(private client: RedisClientType, private ttl: number = 60) { }

  async lookup(key: string): Promise<string | null> {
    try {
      return await this.client.get(`cache:${key}`);
    } catch (error) {
      console.error("Cache lookup failed:", error);
      return null;
    }
  }

  async update(key: string, value: string): Promise<void> {
    try {
      const fullKey = `cache:${key}`;
      console.log("🔑 Attempting to store key:", fullKey);
      console.log("💾 Value length:", value.length);
      console.log("⏰ TTL:", this.ttl);

      await this.client.setEx(fullKey, this.ttl, value);

      // Immediately verify it was stored
      const verification = await this.client.get(fullKey);
      console.log("✅ Verification - key exists:", !!verification);
      console.log("📊 Verification - value length:", verification?.length);

    } catch (error) {
      console.error("Cache update failed:", error);
      throw error;
    }
  }
}

class CacheService {
  private client: RedisClientType;
  private cache: CacheInterface;
  private isConnected: boolean = false;

  constructor() {
    // Initialize Redis client
    this.client = createClient({
      url: process.env.REDIS_URL,
      socket: {
        reconnectStrategy: (retries) => Math.min(retries * 100, 5000) // Exponential backoff
      }
    });

    // Add error handling for the client
    this.client.on('error', (err) => {
      console.error('Redis Client Error:', err);
      this.isConnected = false;
    });

    this.client.on('connect', () => {
      console.log('Redis Client Connected');
      this.isConnected = true;
    });

    this.client.on('disconnect', () => {
      console.log('Redis Client Disconnected');
      this.isConnected = false;
    });

    // Initialize cache adapter
    this.cache = new SimpleCacheAdapter(this.client, 3600);

    this.connect().catch(console.error);
  }

  private async connect(): Promise<void> {
    try {
      if (!this.isConnected) {
        await this.client.connect();
        console.log("Redis connected successfully");
        this.isConnected = true;
      }
    } catch (error) {
      console.error("Redis connection error:", error);
      this.isConnected = false;
      throw error;
    }
  }

  private async ensureConnection(): Promise<void> {
    if (!this.isConnected) {
      await this.connect();
    }
  }

  /** Cache lookup - works with both simple cache and LangChain patterns */
  async lookup(key: string): Promise<string | null> {
    try {
      await this.ensureConnection();
      return await this.cache.lookup(key);
    } catch (error) {
      console.error("Cache lookup failed:", error);
      return null;
    }
  }

  /** Cache update - works with both simple cache and LangChain patterns */
  async update(key: string, value: string): Promise<void> {
    try {
      await this.ensureConnection();
      await this.cache.update(key, value);
    } catch (error) {
      console.error("Cache update failed:", error);
      throw error;
    }
  }

  /** LangChain-compatible lookup for prompt caching */
  async lookupPrompt(prompt: string, llmKey: string = "default"): Promise<string | null> {
    const cacheKey = `${llmKey}:${this.hashPrompt(prompt)}`;
    return await this.lookup(cacheKey);
  }

  /** LangChain-compatible update for prompt caching */
  async updatePrompt(prompt: string, response: string, llmKey: string = "default"): Promise<void> {
    const cacheKey = `${llmKey}:${this.hashPrompt(prompt)}`;
    await this.update(cacheKey, response);
  }

  /** Direct Redis operations */
  async get(key: string): Promise<string | null> {
    try {
      await this.ensureConnection();
      return await this.client.get(key);
    } catch (error) {
      console.error("Redis GET failed:", error);
      return null;
    }
  }

  async set(key: string, value: string, ttl?: number): Promise<void> {
    try {
      await this.ensureConnection();
      if (ttl) {
        await this.client.setEx(key, ttl, value);
      } else {
        await this.client.set(key, value);
      }
    } catch (error) {
      console.error("Redis SET failed:", error);
      throw error;
    }
  }

  async del(key: string): Promise<void> {
    try {
      await this.ensureConnection();
      await this.client.del(key);
    } catch (error) {
      console.error("Redis DEL failed:", error);
      throw error;
    }
  }

  async exists(key: string): Promise<boolean> {
    try {
      await this.ensureConnection();
      const result = await this.client.exists(key);
      return result === 1;
    } catch (error) {
      console.error("Redis EXISTS failed:", error);
      return false;
    }
  }

  async expire(key: string, ttl: number): Promise<void> {
    try {
      await this.ensureConnection();
      await this.client.expire(key, ttl);
    } catch (error) {
      console.error("Redis EXPIRE failed:", error);
      throw error;
    }
  }

  async flushCache(pattern: string = "cache:*"): Promise<void> {
    try {
      await this.ensureConnection();
      const keys = await this.client.keys(pattern);
      if (keys.length > 0) {
        await this.client.del(keys);
      }
    } catch (error) {
      console.error("Cache flush failed:", error);
      throw error;
    }
  }

  /** Simple hash function for prompt keys */
  private hashPrompt(prompt: string): string {
    let hash = 0;
    for (let i = 0; i < prompt.length; i++) {
      const char = prompt.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36);
  }

  /** Check if cache is ready */
  isReady(): boolean {
    return this.isConnected;
  }

  /** Get connection status */
  getStatus(): { connected: boolean; client: string } {
    return {
      connected: this.isConnected,
      client: this.client?.isReady ? 'ready' : 'not ready'
    };
  }

  /** Graceful shutdown */
  async disconnect(): Promise<void> {
    try {
      if (this.isConnected) {
        await this.client.quit();
        console.log("Redis disconnected successfully");
        this.isConnected = false;
      }
    } catch (error) {
      console.error("Redis disconnect error:", error);
    }
  }
}

// Singleton instance
let cacheServiceInstance: CacheService | null = null;

export const getCacheService = async (): Promise<CacheService> => {
  if (!cacheServiceInstance) {
    cacheServiceInstance = new CacheService();

    // Wait for connection to be established
    let attempts = 0;
    const maxAttempts = 50; // 5 seconds total

    while (!cacheServiceInstance.isReady() && attempts < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, 100));
      attempts++;
    }

    if (!cacheServiceInstance.isReady()) {
      throw new Error("Cache service failed to initialize within timeout");
    }
  }
  return cacheServiceInstance;
};

// For backward compatibility
export const cacheService = new CacheService();

// Handle process termination
const gracefulShutdown = async (signal: string) => {
  console.log(`Received ${signal}. Gracefully shutting down...`);
  try {
    if (cacheServiceInstance) {
      await cacheServiceInstance.disconnect();
    }
    await cacheService.disconnect();
  } catch (error) {
    console.error("Error during graceful shutdown:", error);
  }
  process.exit(0);
};

process.on("SIGTERM", () => gracefulShutdown("SIGTERM"));
process.on("SIGINT", () => gracefulShutdown("SIGINT"));

export default CacheService;