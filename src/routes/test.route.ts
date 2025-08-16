// routes/test.route.ts (or wherever you want to create this file)
import express, { Request, Response } from 'express';
import { Document } from "@langchain/core/documents";
import { vectorStoreService } from '../services/vectorstore.service'; // Adjust path as needed

const router = express.Router();

// GET /api/test/add-data - Simple endpoint to add test data
router.get('/add-data', async (req: Request, res: Response) => {
  try {
    console.log("🧪 Adding test data to vector store...");
    
    // Create simple test documents
    const docs = [
      new Document({
        pageContent: "Hello, I am a test document about AI and machine learning.",
        metadata: { source: "test1.txt" }
      }),
      new Document({
        pageContent: "This is another test document about web development and programming.",
        metadata: { source: "test2.txt" }
      }),
      new Document({
        pageContent: "DevUps is a digital agency that helps with web development and digital marketing.",
        metadata: { source: "test3.txt" }
      })
    ];

    // Add to vector store
    await vectorStoreService.addDocuments(docs);
    console.log("✅ Test documents added!");

    res.json({ 
      success: true, 
      message: "Test data added successfully",
      documentsAdded: docs.length
    });

  } catch (error) {
    console.error("❌ Error adding test data:", error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

// GET /api/test/search?q=your_query - Simple endpoint to test search
router.get('/search', async (req: Request, res: Response) => {
  try {
    const query = req.query.q as string || "What is AI?";
    console.log(`🔍 Testing search for: "${query}"`);

    const results = await vectorStoreService.similaritySearchWithScore(query, 3);
    console.log(`✅ Found ${results.length} results`);

    res.json({
      success: true,
      query,
      results: results.map(([doc, score], index) => ({
        rank: index + 1,
        score: Number(score.toFixed(3)),
        source: doc.metadata?.source,
        content: doc.pageContent
      }))
    });

  } catch (error) {
    console.error("❌ Search test failed:", error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

// GET /api/test/status - Check vector store status
router.get('/status', async (req: Request, res: Response) => {
  try {
    const health = await vectorStoreService.healthCheck();
    const count = await vectorStoreService.getDocumentCount();

    res.json({
      success: true,
      vectorStore: {
        initialized: health.initialized,
        connected: health.connected,
        embeddingDimensions: health.embeddingDimensions,
        documentCount: count,
        error: health.error || null
      }
    });

  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

export default router;