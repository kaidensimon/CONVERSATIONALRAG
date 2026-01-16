import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
from .data_loader import load_and_chunk_pdf, embed_texts
from .vector_db import QDrantStorage
from .custom_types import *
from AGENTS.agent import Agent
import os
import websockets
import base64
import json
import asyncio
import os, json, base64, asyncio, logging, inspect
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import time
import requests
from dataclasses import dataclass
from AGENTS.tts import speak, generate_audio
from speechmatics.tts import AsyncClient
from inngest.experimental import ai
import httpx
log = logging.getLogger("uvicorn.error")

load_dotenv()

agent = Agent()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

async_client = httpx.AsyncClient(timeout=5.0)

ASAI_KEY = os.environ["ASAI_KEY"]

# Twilio sends mu-law 8k. Configure AssemblyAI to accept mu-law 8k directly.
ASAI_URL = (
    "wss://streaming.assemblyai.com/v3/ws"
    "?sample_rate=8000"
    "&encoding=pcm_mulaw"
    "&end_of_turn_confidence_threshold=0.4"
    "&min_end_of_turn_silence_when_confident=160"
    "&max_turn_silence=1280"
)




@dataclass
class State:
    streamSID: str = None

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RagUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        inngest_client.logger.info(f"ids: {ids}")
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QDrantStorage().upsert(ids, vecs, payloads)
        return RagUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)

    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RagUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int= 5):
        query_vec = embed_texts([question])[0]
        store = QDrantStorage()
        found = store.search(query_vec, top_k)
        return RagSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embedd-and-search", lambda: _search(question, top_k), output_type=RagSearchResult)
    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question. \n\n" \
        f"Context: \n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above"
    )

    return {"rag_result": user_content}
    
    
@inngest_client.create_function(
    fn_id="LLM: RAG RESPONSE",
    trigger=inngest.TriggerEvent(event="llm/rag_response")
)
async def llm_rag_response(ctx: inngest.Context):
        inngest_client.logger.info("QUERYING LLM")
        history = ctx.event.data["history"]
        adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
        res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": history
        })

        answer = res["choices"][0]["message"]["content"].strip()
        inngest_client.logger.info("LLM QUERY DONE!")
        return {"answer": answer}
    
app = FastAPI()

@app.get("/media")
def media_http():
    return {"status": "WebSocket endpoint ready"}

def _ws_connect_kwargs():
    """
    websockets library renamed `extra_headers` -> `additional_headers` in newer versions.
    This helper supports both.
    """
    params = inspect.signature(websockets.connect).parameters
    if "additional_headers" in params:
        return {"additional_headers": {"Authorization": ASAI_KEY}}
    return {"extra_headers": {"Authorization": ASAI_KEY}}




@app.websocket("/media")
async def media_ws(twilio_ws: WebSocket):
    await twilio_ws.accept()

    # 50ms of mu-law @ 8kHz = 0.05 * 8000 = 400 bytes
    FRAME_BYTES = 400
    buf = bytearray()
    state = State()
    agent = Agent()

    async with websockets.connect(
        ASAI_URL,
        ping_interval=5,
        ping_timeout=60,
        **_ws_connect_kwargs(),
    ) as aai_ws:

        async def twilio_to_aai():
            nonlocal buf
            try:
                while True:
                    raw = await twilio_ws.receive_text()
                    msg = json.loads(raw)

                    ev = msg.get("event")
                    if ev == "media":
                        payload_b64 = msg["media"]["payload"]
                        chunk = base64.b64decode(payload_b64)
                        buf.extend(chunk)

                        # send in ~50ms frames (recommended by AssemblyAI)
                        while len(buf) >= FRAME_BYTES:
                            await aai_ws.send(bytes(buf[:FRAME_BYTES]))
                            del buf[:FRAME_BYTES]

                    elif ev == "stop":
                        break
                    elif ev == "start":
                        state.streamSID = msg["start"]["streamSid"]
                        inngest_client.logger.info(state.streamSID)


                    # ignore "start"/"mark" unless you need them
            except WebSocketDisconnect:
                pass
            finally:
                # flush remainder (optional)
                if buf:
                    await aai_ws.send(bytes(buf))
                    buf.clear()
                # tell AssemblyAI the stream is done
                await aai_ws.send(json.dumps({"type": "Terminate"}))



        async def fetch_runs(event_id: str) -> list[dict]:
            url = f"{_inngest_api_base()}/events/{event_id}/runs"
            resp = await async_client.get(url)
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])

                
                
        def _inngest_api_base() -> str:
            # Local dev server default; configurable via env
            return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")

        async def wait_for_run_output(
        event_id: str,
        timeout_s: float = 15.0,
        poll_interval_s: float = 0.2,  # 200 ms is usually plenty
    ) -> dict:
            start = time.time()
            last_status = None

            while True:
                runs = await fetch_runs(event_id)
                if runs:
                    run = runs[0]
                    status = run.get("status")
                    last_status = status or last_status
                    inngest_client.logger.info(status)

                    if status in ("Completed", "Succeeded", "Success", "Finished"):
                        return run.get("output") or {}

                    if status in ("Failed", "Cancelled"):
                        raise RuntimeError(f"Function run {status}")

                if time.time() - start > timeout_s:
                    raise TimeoutError(
                        f"Timed out waiting for run output (last status: {last_status})"
                    )

                await asyncio.sleep(poll_interval_s)

        
        async def send_rag_query_event(question: str, top_k: int) -> None:
            result = await inngest_client.send(
                inngest.Event(
                    name="rag/query_pdf_ai",
                    data={
                        "question": question,
                        "top_k": top_k,
                    },
                )
            )

            return result[0]
        

    
        async def send_LLM_query_event():
            result = await inngest_client.send(inngest.Event(
                    name="llm/rag_response",
                data={"history": agent.messages.copy()},
                ))
            return result[0]
        

        async def query_rag_no_inngest(question, top_k):
            query_vec = embed_texts([question])[0]
            store = QDrantStorage()
            found = store.search(query_vec, top_k)

            context_block = "\n\n".join(f"- {c}" for c in found["contexts"])
            user_content = (
                "Use the following context to answer the question. \n\n" \
                f"Context: \n{context_block}\n\n"
                f"Question: {question}\n"
                "Answer concisely using the context above"
            )

            return user_content
        
            
        async def talk(answer):
            async for mulaw_frame in agent.speech(text=answer):
                        payload_b64 = base64.b64encode(mulaw_frame).decode("ascii")
                        msg = {
                            "event": "media",                    # REQUIRED: Tells Twilio this is audio
                            "streamSid": state.streamSID,        # REQUIRED: Your stored streamSid
                            "media": {                           # REQUIRED: Media object
                                "payload": payload_b64,          # REQUIRED: base64 μ-law 8000Hz
                            }
                        }

                        await twilio_ws.send_text(json.dumps(msg))    # REQUIRED: send_text(), not send_json()
                        await asyncio.sleep(0.12)


        async def safe_send_event(question: str):
            try:
                #rag_event_id = await send_rag_query_event(question=question, top_k=5)
                #rag_query_output_task = asyncio.create_task(wait_for_run_output(event_id=rag_event_id))
                #rag_output = await rag_query_output_task
                #inngest_client.logger.info("RAG OUTPUT RECEIVED")

                #user_content = rag_output.get("rag_result", "")

                agent.messages.append({"role": "user", "content": question})

                end_turn = False
                while end_turn == False:
                    answer = await agent.invoke()
                    if answer:
                        talk_task = asyncio.create_task(talk(answer=answer.response))
                    if answer.query_rag:
                         user_content = await query_rag_no_inngest(question=question, top_k=5)
                         agent.messages.append({"role": "user", "content": user_content})
                    if answer.end_turn:
                        end_turn = True
                    await asyncio.sleep(0.25)
                    


                #LLM_query_event_id = await send_LLM_query_event()
                #LLM_query_output_task = asyncio.create_task(wait_for_run_output(LLM_query_event_id))
                #LLM_output = await LLM_query_output_task
                #answer = LLM_output.get("answer", "")

                inngest_client.logger.info(f"LLM RESPONSE RECIEVED: {answer}")

                    
                                
            except Exception:
                log.exception("Inngest send failed")



        async def aai_to_log():
            try:
                async for text in aai_ws:
                    data = json.loads(text)
                    if data.get("type") == "Turn" and data.get("transcript"):
                        log.info(
                            "AAI: %s (end=%s conf=%.3f)",
                            data["transcript"],
                            data.get("end_of_turn"),
                            float(data.get("end_of_turn_confidence") or 0.0),
                        )
                    
                        if data.get("end_of_turn") == True:
                            asyncio.create_task(safe_send_event(data["transcript"]))


                            

            except Exception as e:
                log.error("AssemblyAI read error: %s", e)

        await asyncio.gather(twilio_to_aai(), aai_to_log())



inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai, llm_rag_response])