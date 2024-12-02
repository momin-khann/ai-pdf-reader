import {Pinecone} from '@pinecone-database/pinecone';
import {ChatOpenAI, OpenAI} from "@langchain/openai";
import {OpenAIEmbeddings} from "@langchain/openai";
import {RecursiveCharacterTextSplitter} from "@langchain/textsplitters";
import {loadQAStuffChain} from "langchain/chains";
import {Document} from "@langchain/core/documents";
import {StringOutputParser} from "@langchain/core/output_parsers";
import {createStuffDocumentsChain} from "langchain/chains/combine_documents";
import {PineconeStore} from "@langchain/pinecone";
import {PromptTemplate} from "@langchain/core/prompts";

const indexName = 'test-index-01'

const pc = new Pinecone({apiKey: process.env.PINECONE_API_KEY!});

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

const customTemplate = `Use the following pieces of context to answer the question.
Use three sentences maximum and keep the answer as concise as possible.
If you don't know the answer, just say "I don't know, thanks for asking!" and don't try to make up an answer.

{context}

Question: {question}

Answer:`;

export const queryPineconeAndLLM = async (question) => {
  try {
    const llm = new ChatOpenAI({model: "gpt-3.5-turbo-0125", temperature: 0});
    const pineconeIndex = pc.index(indexName);
    const customRagPrompt = PromptTemplate.fromTemplate(customTemplate);

    const vectorStore = await PineconeStore.fromExistingIndex(new OpenAIEmbeddings(), {
      pineconeIndex,
      maxConcurrency: 5,
    });

    const retriever = vectorStore.asRetriever();

    const retrievedDocs = await retriever.invoke(question);
    console.log('Retrieved documents:', retrievedDocs[0].pageContent);

    // Combine the content of the retrieved documents into a single context string
    const context = retrievedDocs.map(doc => doc.metadata.pageContent).join("\n");


    const ragChain = await createStuffDocumentsChain({
      llm,
      prompt: customRagPrompt,
      outputParser: new StringOutputParser(),
    });

    const answer = await ragChain.invoke({
      question,
      context: [new Document({pageContent: context})],
    });

    console.log('Generated answer:', answer);
    return answer;

  } catch (error) {
    console.log(error);
  }
}
