import path from 'path';
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { createPineConeIndex, uploadDataToPinecone } from "@/lib/pinecone";

export async function POST() {
  const filePath = path.resolve(process.cwd(), 'documents/sample.pdf');
  const loader = new PDFLoader(filePath);
  const [doc] = await loader.load();

  try {
    await createPineConeIndex();
    await uploadDataToPinecone(doc);
  } catch (err) {
    console.log('error: ', err);
  }

  return Response.json({
    data: 'successfully created index and loaded data into pinecone...'
  });
}
