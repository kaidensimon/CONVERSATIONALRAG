import logging
from fastapi import FastAPI
from dotenv import load_dotenv
from .data_loader import embed_texts
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
from dataclasses import dataclass
log = logging.getLogger("uvicorn.error")
load_dotenv()


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

                        # send in ~50ms frames
                        while len(buf) >= FRAME_BYTES:
                            await aai_ws.send(bytes(buf[:FRAME_BYTES]))
                            del buf[:FRAME_BYTES]

                    elif ev == "stop":
                        break
                    elif ev == "start":
                        state.streamSID = msg["start"]["streamSid"]


            except WebSocketDisconnect:
                # flush remainder (optional)
                if buf:
                    await aai_ws.send(bytes(buf))
                    buf.clear()
                # tell AssemblyAI the stream is done
                await aai_ws.send(json.dumps({"type": "Terminate"}))



    

       

    

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


        async def safe_send_event(question: str):
            try:

                agent.messages.append({"role": "user", "content": question})

                end_turn = False
                while end_turn == False:
                    answer = await agent.invoke()
                    if answer:
                        await talk(answer=answer.response)
                    if answer.query_rag:
                         user_content = await query_rag_no_inngest(question=question, top_k=5)
                         agent.messages.append({"role": "user", "content": user_content})
                    if answer.end_turn:
                        end_turn = True

                    
                                
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



