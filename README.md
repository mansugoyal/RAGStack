# RAGStack
RAGStack is a modular backend framework for building Retrieval-Augmented Generation (RAG) pipelines using local embeddings, vector search, and LLMs.

## Features
- **Local Embeddings**: Uses HuggingFace-compatible models via [@xenova/transformers](https://www.npmjs.com/package/@xenova/transformers) to generate embeddings directly in Node.js—no API keys or external calls required.
- **Fast & Efficient**: Loads the embedding model once and reuses it for all future requests, ensuring high performance.
- **Easy to Use**: Simple API for converting text to vector embeddings.

## Example Usage

```js
const generateEmbedding = require('./src/embeddings/generateEmbedding');

(async () => {
	const text = "Node.js is fast";
	const embedding = await generateEmbedding(text);
	console.log("Embedding length:", embedding.length);
	console.log("First 5 values:", embedding.slice(0, 5));
})();
```

See [src/index.js](src/index.js) for a runnable example.
