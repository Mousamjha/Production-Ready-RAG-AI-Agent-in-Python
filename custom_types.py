import pydantic

class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str = None


class RAGUpsertResult(pydantic.BaseModel):
    ingested: int


class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: set[str]


class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: set[str]
    num_contexts: int