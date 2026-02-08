from typing import AsyncGenerator
from speechmatics.tts import AsyncClient, Voice, OutputFormat
import asyncio
from dotenv import load_dotenv
import logging
import asyncio
from speechmatics.tts import AsyncClient, Voice, OutputFormat
load_dotenv()
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # int16 -> 2 bytes
CHUNK_SIZE = 1024  
BYTES_PER_CHUNK = CHUNK_SIZE * CHANNELS * SAMPLE_WIDTH
BUFFER_SIZE = 4096
END_OF_STREAM = None
OUTPUT_FORMAT = OutputFormat.RAW_PCM_16000

logger = logging.getLogger(__name__)

async def generate_audio(text: str, audio_queue: asyncio.Queue, voice: Voice, client: AsyncClient) -> None:
        async with await client.generate(
            text=text,
            voice=voice,
            output_format=OutputFormat.RAW_PCM_16000
        ) as response:
            buf = bytearray()

            async for chunk in response.content.iter_chunked(BUFFER_SIZE):
                #logger.info(len(chunk))
                if not chunk:
                    continue

                buf.extend(chunk)

                # Emit full CHUNK_SIZE frames worth of PCM bytes
                while len(buf) >= BYTES_PER_CHUNK:
                    out = bytes(buf[:BYTES_PER_CHUNK])
                    del buf[:BYTES_PER_CHUNK]
                    await audio_queue.put(out)

            # Flush remainder (keep sample alignment)
            remainder = len(buf) - (len(buf) % SAMPLE_WIDTH)
            if remainder > 0:
                await audio_queue.put(bytes(buf[:remainder]))

            await audio_queue.put(END_OF_STREAM)

            
async def speak(play_queue: asyncio.Queue) -> AsyncGenerator[bytes, None]:
    while True:
        item = await play_queue.get()
        try:
            if item is END_OF_STREAM:
                break
            yield item
        finally:
            play_queue.task_done()
