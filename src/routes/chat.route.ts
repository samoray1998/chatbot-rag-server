import express from 'express';
import { llmService } from '../services/llm.service';

const router = express.Router();

router.post('/', async (req, res) => {
  try {
    const { message } = req.body;
    if (!message) {
      return res.status(400).json({ error: "Message is required" });
    }

    //const response = await llmService.generate(message);
    const response= await llmService.generateWithContext(message)
    res.json({ response });
  } catch (error) {
    console.error("Chat error:", error);
    res.status(500).json({ error: "Failed to generate response" });
  }
});

export default router;