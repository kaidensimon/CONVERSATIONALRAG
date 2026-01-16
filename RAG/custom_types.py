import pydantic


class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str = None

class RagUpsertResult(pydantic.BaseModel):
    ingested: int


class RagSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]

class RagQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int


