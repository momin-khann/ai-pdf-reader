import { NextRequest, NextResponse } from 'next/server'
import {queryPineconeAndLLM} from "@/lib/pinecone";

export async function POST(req: NextRequest) {
  const question = await req.json();
  console.log("---- ", question)

  const text = await queryPineconeAndLLM(question)

  return NextResponse.json({
    data: text
  })
}
