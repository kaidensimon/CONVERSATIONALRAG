import asyncio
from .tts import speak, generate_audio
from speechmatics.tts import AsyncClient
from .convert_audio import Pcm16kToMulaw8k
import inngest
from speechmatics.tts import Voice
from langchain_openai import ChatOpenAI
from .custom_types import AgentOutput
from .system_prompts import AGENT_PROMPT
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)



class Agent:
    def __init__(self):
        
        self.VOICE = Voice.SARAH
        self.messages = [{"role": "system", "content": AGENT_PROMPT}]
        self.tts_client = AsyncClient()
        self.queue = asyncio.Queue()
        self.inngest_client = get_inngest_client()
        self.llm = ChatOpenAI(model="gpt-4o").with_structured_output(AgentOutput)


    async def invoke(self):
        response: AgentOutput = await self.llm.ainvoke(self.messages)
        self.messages.append({"role": "ai", "content": response.response})
        return response
    
    async def speech(self, text: str):
        producer = asyncio.create_task(generate_audio(text, self.queue, self.VOICE, self.tts_client))
        # Use 50ms μ-law frames (400 bytes @8k) to match AssemblyAI/Twilio limits.
        conv = Pcm16kToMulaw8k(frame_ms=50)

        try:
            async for pcm_chunk_16k in speak(self.queue):
                # pcm_chunk_16k is int16 PCM @ 16kHz
                for mulaw_frame in conv.feed(pcm_chunk_16k):
                    # mulaw_frame is 20ms μ-law @ 8kHz (160 bytes)
                    yield mulaw_frame

            # End-of-stream: optionally flush + pad last partial frame
            for mulaw_frame in conv.flush(pad_to_full_frame=True):
                yield mulaw_frame

        except Exception as e:
            print(e)

        finally:
            producer.cancel()
    

    async def close_tts_client(self):
        await self.tts_client.close()
