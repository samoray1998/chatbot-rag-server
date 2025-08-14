// src/validations/chat.ts
import { z } from "zod";

export const ChatMessageSchema = z.object({
  type: z.enum(["human", "ai"]),
  content: z.string().min(1).max(2000)
});

export const ChatRequestSchema = z.object({
  question: z.string().min(1).max(1000),
  history: z.array(ChatMessageSchema).max(20) // Limit to 20 messages in history
});

export type ChatRequest = z.infer<typeof ChatRequestSchema>;