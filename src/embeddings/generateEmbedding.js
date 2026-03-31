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

let embedder;

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
  const model = await getEmbedder();

  const output = await model(text, {
    pooling: "mean",      // ✅ IMPORTANT (handles pooling for you)
    normalize: true       // ✅ gives better similarity results
  });

  return Array.from(output.data); // convert Float32Array → normal array
}

module.exports = generateEmbedding;