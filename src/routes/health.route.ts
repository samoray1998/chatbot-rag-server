import express from 'express';
import { cacheService } from '../services/cache.service';
import { llmService } from '../services/llm.service';

const router = express.Router();

router.get('/', async (req, res) => {
  try {
    // Test cache
    await cacheService.set('healthcheck', 'ok');
    const cacheStatus = await cacheService.get('healthcheck');
    
    // Test LLM (simple ping)
    await llmService.generate("ping");
    
    res.json({ 
      status: 'healthy',
      cache: cacheStatus === 'ok' ? 'working' : 'broken',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(503).json({ 
      status: 'unhealthy',
      error: error 
    });
  }
});

export default router;