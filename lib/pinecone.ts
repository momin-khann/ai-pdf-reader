import {Pinecone} from '@pinecone-database/pinecone';
import {OpenAI} from "@langchain/openai";
import {OpenAIEmbeddings} from "@langchain/openai";
import {RecursiveCharacterTextSplitter} from "@langchain/textsplitters";
import {createStuffDocumentsChain} from "langchain/chains/combine_documents";
import {loadQAStuffChain} from "langchain/chains";
import { Document } from "@langchain/core/documents";

const indexName = 'test-index-01'

const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });

export const createPineConeIndex = async () => {
  try {
    const {indexes} = await pc.listIndexes();

    // @ts-ignore
    if (!indexes.includes(indexName)) {
      await pc.createIndex({
        name: indexName,
        dimension: 1536,
        metric: 'cosine',
        spec: {
          serverless: {
            cloud: 'aws',
            region: 'us-east-1'
          }
        }
      });
    }
  } catch (error) {
    console.log(error);
  }
}

export const uploadDataToPinecone = async (doc) => {
  try {
    // 1. Retrieve Pinecone index
    const index = pc.index(indexName);
    // 2. Log the retrieved index name
    console.log(index);

    // 3. Process document in the docs array
    console.log(`Processing document: ${doc.metadata.source}`);
    const textPath = doc.metadata.source;
    const text = doc.pageContent;

    // 4. Create RecursiveCharacterTextSplitter instance
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 100,
      chunkOverlap: 0,
    });

    // 5. Split text into chunks (documents)
    console.log("splitting text into chunks....");
    const chunks = await textSplitter.createDocuments([text]);
    console.log(`text split into ${chunks.length} chunks`);

    // 6. Create OpenAI embeddings for documents
    console.log("calling OpenAI's embedding endpoint");
    const embeddingVector = await new OpenAIEmbeddings().embedDocuments(chunks.map(chunk => chunk.pageContent.replace(/\n/g, " ")));
    console.log('Finished embedding documents');

    // 7. Create and upsert vectors in batches of 100
    const batchSize = 100;
    let batch = [];
    for (let idx = 0; idx < chunks.length; idx++) {
      const chunk = chunks[idx];
      const vector = {
        id: `${textPath}_${idx}`,
        values: embeddingVector[idx],
        metadata: {
          ...chunk.metadata,
          loc: JSON.stringify(chunk.metadata.loc),
          pageContent: chunk.pageContent,
          txtPath: textPath,
        },
      };
      batch = [...batch, vector]
      // When batch is full or it's the last item, upsert the vectors
      if (batch.length === batchSize || idx === chunks.length - 1) {
        await index.upsert(batch);

        // Empty the batch
        batch = [];
      }
    }


    // 8. Log the number of vectors updated
    console.log(`Pinecone index updated`);
  } catch (error) {
    console.log(error);
  }
}

export const queryPineconeAndLLM = async (question) => {
  try {
    // 1. Start query process
    console.log('Querying Pinecone vector store...');
    // 2. Retrieve the Pinecone index
    const index = pc.index(indexName);

    // 3. Create embedding Query
    const queryEmbedding = await new OpenAIEmbeddings().embedQuery(question);

    // 4. Query Pinecone index and return top 10 matches
    let queryResponse = await index.query({
        topK: 10,
        vector: queryEmbedding,
        includeMetadata: true,
        includeValues: true,
    });

    // 5. Log the number of matches
    console.log(`Found ${queryResponse.matches.length} matches...`);

    // 6. Log the question being asked
    console.log(`Asking question: ${question}...`);

    if (queryResponse.matches.length) {
      // 7. Create an OpenAI instance and load the QAStuffChain
      const model = new OpenAI({});
      const chain = loadQAStuffChain(model);
      // 8. Extract and concatenate page content from matched documents
      const concatenatedPageContent = queryResponse.matches
        .map((match) => match.metadata.pageContent)
        .join(" ");
      // 9. Execute the chain with input documents and question
      const result = await chain.call({
        input_documents: [new Document({ pageContent: concatenatedPageContent })],
        question: question,
      });
      // 10. Log the answer
      console.log(`Answer: ${result.text}`);
      return result.text
    } else {
      // 11. Log that there are no matches, so GPT-3 will not be queried
      console.log('Since there are no matches, GPT-3 will not be queried.');
    }

  } catch (error) {
    console.log(error);
  }
}
