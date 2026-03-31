require("dotenv").config();
const generateEmbedding = require("./embeddings/generateEmbedding");

async function main() {
    const text = "Node.js is fast";

    const embedding = await generateEmbedding(text);

    console.log("Embedding length:", embedding.length);
    console.log("First 5 values:", embedding.slice(0, 5));
    const emb1 = await generateEmbedding("Node.js is fast");
    const emb2 = await generateEmbedding("Node.js is high performance");
    const emb3 = await generateEmbedding("I love pizza but dominos is not my favorite");

    console.log("Embedding length 1:", emb1.length);
    console.log("Embedding length 2:", emb2.length);
    console.log("Embedding length 3:", emb3.length);
}

main();