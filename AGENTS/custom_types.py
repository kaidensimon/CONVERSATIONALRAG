from pydantic import BaseModel, Field

class AgentOutput(BaseModel):
    response: str = Field(..., description="The agent's textual response")
    end_turn: bool = Field(default=False, description="Whether or not you are done with your turn.")
    query_rag: bool = Field(default=False, description="Whether or not you need to query your knowledge base.")