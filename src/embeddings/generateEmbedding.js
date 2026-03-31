// Using OpenAI's API for embedding generation
// const { OpenAI } = require("openai");
// const client = new OpenAI({
//   apiKey: process.env.OPENAI_API_KEY
// });
// async function generateEmbedding(text) {
//   const response = await client.embeddings.create({
//     model: "text-embedding-3-small",
//     input: text
//   });
//   return response.data[0].embedding;
// }
// module.exports = generateEmbedding;

const { pipeline } = require("@xenova/transformers");
// This is HuggingFace-style transformer running locally in Node.js

let embedder;

// 👉 What’s happening:
// Model loads only once
// Reused for all future calls

// 💡 Why important:
// Model loading is slow (~seconds)
// Avoid reloading → huge performance gain

async function getEmbedder() {
    if (!embedder) {
        embedder = await pipeline(
            "feature-extraction",
            "Xenova/all-MiniLM-L6-v2"
        );
    }
    return embedder;
}

async function generateEmbedding(text) {
    // 🧠 Step 1: Get model
    const model = await getEmbedder();

    // 🧠 Step 2: Run inference
    const output = await model(text, {
        pooling: "mean",      // ✅ IMPORTANT (handles pooling for you)
        normalize: true       // ✅ gives better similarity results
    });

    // ✅ pooling: "mean"
    // Transformer outputs vectors per token
    // This combines them into single vector

    // Without pooling:
    // [ [token1], [token2], [token3] ]

    // With pooling:
    // [final embedding vector]

    // ✅ normalize: true
    // Makes vector unit length
    // Required for cosine similarity

    // 💡 Why:
    // Improves similarity accuracy
    // Standard practice in vector search

    return Array.from(output.data); // convert Float32Array → normal array
}

module.exports = generateEmbedding;