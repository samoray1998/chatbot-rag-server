// src/config/qdrant.ts
import { QdrantClient } from "@qdrant/js-client-rest";
import dotenv from "dotenv";

dotenv.config();

const QDRANT_URL = process.env.QDRANT_URL || "http://qdrant:6333";


export const qdrantClient = new QdrantClient({
  url: QDRANT_URL,
});
