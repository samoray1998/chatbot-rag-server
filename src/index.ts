import express from 'express';
import chatRouter from './routes/chat.route';
import healthRouter from './routes/health.route';
import testRouter from './routes/test.route';

const app = express();
app.use(express.json());

// Routes
app.use('/api/chat', chatRouter);
app.use('/health', healthRouter);

app.use('/api/test', testRouter);


// Error handling
app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error(err.stack);
  res.status(500).send('Something broke!');
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});