# from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
EMBED_MODEL = "gemini-embedding-2"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)

def load_and_chunk(path: str):
    docs = PDFReader().load_data(file = path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:

    contents = [
        types.Content(
            parts=[
        types.Part.from_text(
            text=text
            )
        ]
    )
    for text in texts
    ]

    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=contents,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBED_DIM
        )
    )
    return [
        embedding.values for embedding in response.embeddings
    ]