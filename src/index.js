require("dotenv").config();
const generateEmbedding = require("./embeddings/generateEmbedding");

async function main() {
    const text = "Node.js is fast";
    // Converts text → vector (array of numbers))
    const embedding = await generateEmbedding(text);
    // Example : [0.12, -0.45, 0.89, ...]

    console.log("Embedding length:", embedding.length);
    console.log("First 5 values:", embedding.slice(0, 5));
    // Length = fixed size vector (e.g., 384)
    // Same model → same length always

    const emb1 = await generateEmbedding("Node.js is fast");
    const emb2 = await generateEmbedding("Node.js is high performance");
    const emb3 = await generateEmbedding("I love pizza but dominos is not my favorite");

    console.log("Embedding length 1:", emb1.length);
    console.log("Embedding length 2:", emb2.length);
    console.log("Embedding length 3:", emb3.length);
    // Similar sentences → similar embeddings
    // Different sentences → very different embeddings
}

// This is entry point to test embeddings.
main();