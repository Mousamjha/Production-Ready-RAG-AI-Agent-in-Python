import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult
from pathlib import Path
from google import genai
from google.genai import types

gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)

async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context):
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}: {i}")) for i in range(len(chunks))]      
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]

        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run(
        "Load and Chunk PDF",
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc
    )

    ingested = await ctx.step.run(
        "Upsert Chunks to Vector DB",
         lambda: _upsert(chunks_and_src),
        output_type= RAGUpsertResult
    )
    return ingested.model_dump()


# @inngest_client.create_function(
#     fn_id = "RAG Query PDF",
#     trigger = inngest.TriggerEvent(event="rag/query_pdf_ai") 
# )
# async def rag_query_pdf_ai(ctx: inngest.Context):
#     def _search(question: str, top_k: int = 5) -> RAGSearchResult:
#         query_vec = embed_texts([question])[0]
#         store = QdrantStorage()
#         found = store.search(query_vec, top_k)
#         return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

#     question = ctx.event.data["question"]
#     top_k = int(ctx.event.data.get("top_k", 5))

#     found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

#     context_block = "\n\n".join(f" - {c}" for c in found.contexts)
#     user_content = (
#         "Use the following context to answer the question.\n\n"
#         f"Context: \n{context_block}\n\n"
#         f"Question: {question}\n"
#         "Answer concisely using the context above."
#     )

#     adapter = ai.openai.Adapter(
#         auth_key=os.getenv("OPENAI_API_KEY"),
#         model="gpt-4o-mini"
#     )

#     res = await ctx.step.ai.infer(
#         "llm-answer",
#         adapter=adapter,
#         body={
#             "max_tokens": 1024,
#             "temperature": 0.2,
#             "messages": [
#                 {"role": "system", "content":"You answer questions using only the provided context."},
#                 {"role": "user", "content":user_content},
#             ]
#         }
#     )

#     answer = res["choices"][0]["messages"]["content"].strip()
#     return {
#         "answer": answer,
#         "sources": found.sources,
#         "num_context": len(found.contexts)
#     }

@inngest_client.create_function(
    fn_id="RAG Query PDF",
    trigger=inngest.TriggerEvent(
        event="rag/query_pdf_ai"
    )
)
async def rag_query_pdf_ai(ctx: inngest.Context):

    def _search(
        question: str,
        top_k: int = 5
    ) -> RAGSearchResult:

        query_vec = embed_texts([question])[0]

        store = QdrantStorage()

        found = store.search(
            query_vec,
            top_k
        )

        return RAGSearchResult(
            contexts=found["contexts"],
            sources=list(found["sources"])
        )

    def _generate_answer(
        user_content: str
    ) -> str:

        response = gemini_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You answer questions using only "
                    "the provided context."
                ),
                temperature=0.2,
                max_output_tokens=1024
            )
        )

        return response.text or ""

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    # Search
    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult
    )

    # Create context
    context_block = "\n\n".join(
        f"- {context}"
        for context in found.contexts
    )

    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely using only the context above."
    )

    # Gemini
    answer = await ctx.step.run(
        "llm-answer",
        lambda: _generate_answer(user_content)
    )

    # Everything here is JSON serializable
    return {
        "answer": str(answer),
        "sources": list(found.sources),
        "num_context": len(found.contexts)
    }

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, functions=[rag_ingest_pdf, rag_query_pdf_ai])
