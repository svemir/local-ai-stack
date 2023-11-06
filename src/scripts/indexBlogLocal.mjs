// Call embeding API and insert to supabase
// Ref: https://js.langchain.com/docs/modules/indexes/vector_stores/integrations/supabase

import dotenv from "dotenv";
import { Document } from "langchain/document";
import { createClient } from "@supabase/supabase-js";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { pipeline } from "@xenova/transformers";

import fs from "fs";
import path from "path";

dotenv.config({ path: `.env.local` });

const fileNames = fs.readdirSync("blogs");
const splitter = RecursiveCharacterTextSplitter.fromLanguage("markdown", {
  chunkSize: 1000,
  chunkOverlap: 50,
});

let langchainDocs = await Promise.all(
  fileNames.map(async (fileName) => {
    const filePath = path.join("blogs", fileName);
    const fileContent = fs.readFileSync(filePath, "utf8");
    const splitDocs = await splitter.splitText(fileContent);
    return splitDocs.map((doc) => {
      return new Document({
        metadata: { fileName },
        pageContent: doc,
      });
    });
  })
);

const auth = {
  detectSessionInUrl: false,
  persistSession: false,
  autoRefreshToken: false,
};

const client = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_PRIVATE_KEY,
  { auth }
);

for (let i = 0; i < langchainDocs.length; i++) {
  console.log(langchainDocs[i][0].metadata.fileName);
  let embeddingPromises = langchainDocs[i].map((doc) => {
    return generateEmbedding(doc.pageContent);
  });
  
  let returnedEmbeddings = await Promise.all(embeddingPromises);

  let insertData = [];
  langchainDocs[i].map((doc, index) => {
    insertData.push({
      content: doc.pageContent,
      embedding: returnedEmbeddings[index],
      metadata: doc.metadata,
    });
  });
  let { error } = await client.from("documents").insert(insertData);
  console.debug(error);
}


export async function generateEmbedding(content) {
  const generateEmbedding = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );

  // Generate a vector using Transformers.js
  const output = await generateEmbedding(content, {
    pooling: "mean",
    normalize: true,
  });

  // Extract the embedding output
  const embedding = Array.from(output.data);
  return embedding;
}
