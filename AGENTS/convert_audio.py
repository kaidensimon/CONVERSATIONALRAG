import audioop
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Pcm16kToMulaw8k:
    in_rate: int = 16000
    out_rate: int = 8000
    channels: int = 1
    width: int = 2          # int16 = 2 bytes
    frame_ms: int = 20      # typical telephony frame

    def __post_init__(self):
        # audioop.ratecv state must be carried across chunks
        self._rate_state: Optional[object] = None
        self._carry = b""  # leftover bytes not aligned to sample width
        self._mulaw_buf = bytearray()

        # 20ms at 8kHz = 8000 * 0.02 = 160 samples => 160 bytes in μ-law
        self._out_frame_bytes = int(self.out_rate * self.frame_ms / 1000)

        # μ-law "silence" byte (0 amplitude)
        self._silence = audioop.lin2ulaw(b"\x00\x00", self.width)  # usually b"\xff"

    def feed(self, pcm16_16k: bytes) -> List[bytes]:
        """
        Accepts little-endian int16 PCM at 16kHz.
        Returns a list of μ-law frames (bytes) at 8kHz, each of size _out_frame_bytes.
        """
        if not pcm16_16k:
            return []

        data = self._carry + pcm16_16k

        # Ensure sample alignment (multiple of 2 bytes for int16)
        n = len(data) - (len(data) % self.width)
        self._carry = data[n:]
        data = data[:n]
        if not data:
            return []

        # Downsample 16k -> 8k (still int16 PCM)
        pcm16_8k, self._rate_state = audioop.ratecv(
            data, self.width, self.channels, self.in_rate, self.out_rate, self._rate_state
        )

        # Convert linear PCM -> μ-law (8-bit)
        mulaw = audioop.lin2ulaw(pcm16_8k, self.width)
        self._mulaw_buf.extend(mulaw)

        out: List[bytes] = []
        while len(self._mulaw_buf) >= self._out_frame_bytes:
            out.append(bytes(self._mulaw_buf[:self._out_frame_bytes]))
            del self._mulaw_buf[:self._out_frame_bytes]

        return out

    def flush(self, pad_to_full_frame: bool = False) -> List[bytes]:
        """
        Flush any buffered μ-law data.
        If pad_to_full_frame=True, pads the last partial frame with μ-law silence.
        """
        out: List[bytes] = []
        if not self._mulaw_buf:
            return out

        if pad_to_full_frame and len(self._mulaw_buf) % self._out_frame_bytes != 0:
            missing = self._out_frame_bytes - (len(self._mulaw_buf) % self._out_frame_bytes)
            self._mulaw_buf.extend(self._silence * missing)

        # Emit remaining bytes as frames
        while len(self._mulaw_buf) >= self._out_frame_bytes:
            out.append(bytes(self._mulaw_buf[:self._out_frame_bytes]))
            del self._mulaw_buf[:self._out_frame_bytes]

        # If any remainder still exists, you can drop it or send it (most telephony expects fixed sizes)
        self._mulaw_buf.clear()
        self._carry = b""
        return out
