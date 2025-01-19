import copy
import json
import queue
import re
import sys
import threading
import time
import pickle
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import requests
import sounddevice as sd
import yaml
from jinja2 import Template
from Levenshtein import distance
from loguru import logger
from sounddevice import CallbackFlags

# Vector DB & Embeddings
import faiss
from sentence_transformers import SentenceTransformer
from collections import deque

# Local glados modules (adjust your import paths as needed)
from glados import asr, tts, vad
from glados.llama import LlamaServer, LlamaServerConfig

logger.remove()
logger.add(sys.stderr, level="INFO")

ASR_MODEL = "ggml-medium-32-2.en.bin"
VAD_MODEL = "silero_vad.onnx"

LLM_STOP_SEQUENCE = "<|eot_id|>"

LLAMA3_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"
)

PAUSE_TIME = 0.05
SAMPLE_RATE = 16000
VAD_SIZE = 50
VAD_THRESHOLD = 0.9
BUFFER_SIZE = 600
PAUSE_LIMIT = 905
SIMILARITY_THRESHOLD = 2

# If TTS is still crashing, reduce chunk size further:
CHUNK_SIZE = 100

DEFAULT_PERSONALITY_PREPROMPT = {
    "role": "system",
    "content": (
        "You are a helpful AI assistant. You are here to assist the user in their tasks, "
        "and inform them in depth."
    ),
}


@dataclass
class GladosConfig:
    completion_url: str
    api_key: Optional[str]
    wake_word: Optional[str]
    announcement: Optional[str]
    personality_preprompt: List[dict[str, str]]
    interruptible: bool
    voice_model: str = "glados.onnx"
    speaker_id: int = None

    @classmethod
    def from_yaml(cls, path: str, key_to_config: Sequence[str] | None = ("Glados",)):
        key_to_config = key_to_config or []
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        config = data
        for nested_key in key_to_config:
            config = config[nested_key]
        return cls(**config)


# =========================================
# Memory Module with FAISS-based RAG
# =========================================
class MemoryModule:
    def __init__(self, max_size: int = 10000, index_type: str = "flat"):
        self.max_size = max_size
        self.conversation_memory: deque[str] = deque(maxlen=max_size)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.faiss_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Use different FAISS index types based on size needs
        if index_type == "flat":
            self.faiss_index = faiss.IndexFlatIP(self.faiss_dim)
        elif index_type == "ivf":
            # IVF index for faster search with large datasets
            nlist = min(max(self.max_size // 50, 4), 100)  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.faiss_dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, self.faiss_dim, nlist)
            self.faiss_index.nprobe = min(nlist, 10)  # Number of clusters to search
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.id_map = {}
        self.next_id = 0
        
        # Add cache for embeddings
        self.embedding_cache = {}
        self.cache_size = 1000
        
        # Add batching for embeddings
        self.batch_size = 32
        self.pending_texts = []
        self.pending_ids = []

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        emb = self.embedding_model.encode([text])[0].astype("float32")
        
        # Update cache with LRU strategy
        if len(self.embedding_cache) >= self.cache_size:
            self.embedding_cache.pop(next(iter(self.embedding_cache)))
        self.embedding_cache[text] = emb
        
        return emb

    def _process_batch(self):
        """Process pending texts in batch."""
        if not self.pending_texts:
            return
            
        # Compute embeddings in batch
        embeddings = self.embedding_model.encode(self.pending_texts).astype("float32")
        
        # Train index if using IVF and not trained
        if isinstance(self.faiss_index, faiss.IndexIVFFlat) and not self.faiss_index.is_trained:
            self.faiss_index.train(embeddings)
        
        # Add to index
        self.faiss_index.add(embeddings)
        
        # Update cache
        for text, emb in zip(self.pending_texts, embeddings):
            if len(self.embedding_cache) >= self.cache_size:
                self.embedding_cache.pop(next(iter(self.embedding_cache)))
            self.embedding_cache[text] = emb
        
        # Update id map
        for id_, text in zip(self.pending_ids, self.pending_texts):
            self.id_map[id_] = text
        
        # Clear pending
        self.pending_texts.clear()
        self.pending_ids.clear()

    def add_memory(self, text: str):
        """Add memory with batching support."""
        text = text.strip()
        if not text:
            return
            
        self.conversation_memory.append(text)
        self.pending_texts.append(text)
        self.pending_ids.append(self.next_id)
        self.next_id += 1
        
        # Process batch if full
        if len(self.pending_texts) >= self.batch_size:
            self._process_batch()

    def get_relevant_memories(self, query: str, k: int = 3) -> List[str]:
        """Get relevant memories with optimized search."""
        # Process any pending memories first
        self._process_batch()
        
        if self.faiss_index.ntotal == 0:
            return []
            
        # Search with cached embedding
        query_emb = self._get_embedding(query).reshape(1, -1)
        scores, idxs = self.faiss_index.search(query_emb, k)
        
        # Filter by similarity threshold
        MIN_SCORE = 0.5  # Adjust based on your needs
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or score < MIN_SCORE:
                continue
            if idx in self.id_map:
                results.append(self.id_map[idx])
        
        return results

    def to_dict(self) -> dict:
        """Save state with additional metadata."""
        # Process any pending memories
        self._process_batch()
        
        return {
            "conversation": list(self.conversation_memory),
            "id_map": self.id_map,
            "next_id": self.next_id,
            "index_config": {
                "type": "ivf" if isinstance(self.faiss_index, faiss.IndexIVFFlat) else "flat",
                "dim": self.faiss_dim
            }
        }

    def from_dict(self, data: dict):
        """Load state with optimizations."""
        self.conversation_memory = deque(data.get("conversation", []), maxlen=self.max_size)
        self.id_map = data.get("id_map", {})
        self.next_id = data.get("next_id", 0)
        
        # Recreate index with saved config
        index_config = data.get("index_config", {"type": "flat", "dim": self.faiss_dim})
        if index_config["type"] == "ivf":
            nlist = min(max(len(self.conversation_memory) // 50, 4), 100)
            quantizer = faiss.IndexFlatIP(index_config["dim"])
            self.faiss_index = faiss.IndexIVFFlat(quantizer, index_config["dim"], nlist)
            self.faiss_index.nprobe = min(nlist, 10)
        else:
            self.faiss_index = faiss.IndexFlatIP(index_config["dim"])
        
        # Batch process all memories
        texts = list(self.conversation_memory)
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self.embedding_model.encode(batch).astype("float32")
            
            # Train IVF index if needed
            if isinstance(self.faiss_index, faiss.IndexIVFFlat) and not self.faiss_index.is_trained:
                self.faiss_index.train(embeddings)
            
            self.faiss_index.add(embeddings)
            
            # Update cache
            for text, emb in zip(batch, embeddings):
                if len(self.embedding_cache) >= self.cache_size:
                    self.embedding_cache.pop(next(iter(self.embedding_cache)))
                self.embedding_cache[text] = emb


# =========================================
# Helper: sanitize + chunk text for TTS
# =========================================
def sanitize_for_tts(raw_text: str) -> str:
    """
    1. Replace triple-dots or repeated dots with single.
    2. Remove quotes and parentheses.
    3. Only keep ASCII.
    """
    text = raw_text
    # remove repeated punctuation like "..." or "!!!"
    text = re.sub(r"(\.\.\.)+", ".", text)
    text = re.sub(r"[!?]{2,}", "!", text)

    # remove quotes, parentheses, or other suspicious chars:
    text = re.sub(r'["“”‘’]', '', text)
    text = re.sub(r'[()<>]', '', text)

    # ensure ascii
    text = text.encode("ascii", errors="replace").decode("ascii", errors="replace")
    return text


def chunk_text(raw_text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Break text into small pieces so TTS won't segfault on large strings.
    We'll do a simple word-based chunking approach.
    """
    safe_text = sanitize_for_tts(raw_text)
    words = safe_text.split()
    chunks = []
    current_list = []
    current_len = 0
    for w in words:
        wlen = len(w)
        if (current_len + wlen + 1) >= chunk_size:
            chunk_str = " ".join(current_list)
            if chunk_str.strip():
                chunks.append(chunk_str)
            current_list = [w]
            current_len = wlen
        else:
            current_list.append(w)
            current_len += wlen + 1
    if current_list:
        final_chunk = " ".join(current_list)
        if final_chunk.strip():
            chunks.append(final_chunk)
    return chunks


class Glados:
    def __init__(
        self,
        voice_model: str,
        speaker_id: int,
        completion_url: str,
        api_key: str | None = None,
        wake_word: str | None = None,
        personality_preprompt: Sequence[dict[str, str]] = (DEFAULT_PERSONALITY_PREPROMPT,),
        announcement: str | None = None,
        interruptible: bool = True,
    ):
        self.completion_url = completion_url
        self.wake_word = wake_word
        self.interruptible = interruptible

        self.memory_module = MemoryModule()

        self._vad_model = vad.VAD(model_path=str(Path.cwd() / "models" / VAD_MODEL))
        self._asr_model = asr.ASR(model=str(Path.cwd() / "models" / ASR_MODEL))
        self._tts = tts.Synthesizer(
            model_path=str(Path.cwd() / "models" / voice_model),
            use_cuda=False,
            speaker_id=speaker_id,
        )

        self.prompt_headers = {"Authorization": api_key or "Bearer your_api_key_here"}

        self._sample_queue: queue.Queue[Tuple[np.ndarray, bool]] = queue.Queue()
        self._buffer: queue.Queue[np.ndarray] = queue.Queue(maxsize=BUFFER_SIZE // VAD_SIZE)
        self._samples: List[np.ndarray] = []
        self._recording_started = False
        self._gap_counter = 0

        self.llm_queue: queue.Queue[str] = queue.Queue()
        self.tts_queue: queue.Queue[str] = queue.Queue()

        self.processing = False
        self.currently_speaking = False
        self.shutdown_event = threading.Event()

        self._messages = list(personality_preprompt)
        self.template = Template(LLAMA3_TEMPLATE)

        self.last_user_input_time: float = 0.0

        llm_thread = threading.Thread(target=self.process_LLM)
        llm_thread.start()
        tts_thread = threading.Thread(target=self.process_TTS_thread)
        tts_thread.start()

        if announcement:
            audio = self._tts.generate_speech_audio(announcement)
            logger.success(f"TTS text: {announcement}")
            sd.play(audio, self._tts.rate)
            if not self.interruptible:
                sd.wait()

        def audio_callback_for_sdInputStream(indata: np.ndarray, frames: int, time_info: Any, status: CallbackFlags):
            data = indata.copy().squeeze()
            vad_confidence = self._vad_model.process_chunk(data) > VAD_THRESHOLD
            self._sample_queue.put((data, vad_confidence))

        self.input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=audio_callback_for_sdInputStream,
            blocksize=int(SAMPLE_RATE * VAD_SIZE / 1000),
        )

    @property
    def messages(self) -> Sequence[dict[str, str]]:
        return self._messages

    @classmethod
    def from_config(cls, config: GladosConfig):
        personality_preprompt = []
        for line in config.personality_preprompt:
            personality_preprompt.append({"role": list(line.keys())[0], "content": list(line.values())[0]})
        return cls(
            voice_model=config.voice_model,
            speaker_id=config.speaker_id,
            completion_url=config.completion_url,
            api_key=config.api_key,
            wake_word=config.wake_word,
            personality_preprompt=personality_preprompt,
            announcement=config.announcement,
            interruptible=config.interruptible,
        )

    @classmethod
    def from_yaml(cls, path: str):
        return cls.from_config(GladosConfig.from_yaml(path))

    # -------------------------------------------------------------------------
    # +++ State +++
    # -------------------------------------------------------------------------
    def save_state(self, filename: str = "lumina_state.pkl"):
        data = {
            "messages": self._messages,
            "memory": self.memory_module.to_dict(),
        }
        try:
            with open(filename, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"[Glados] State saved to {filename}.")
        except Exception as e:
            logger.error(f"[Glados] Error saving state: {e}")

    def load_state(self, filename: str = "lumina_state.pkl"):
        try:
            with open(filename, "rb") as f:
                state_data = pickle.load(f)
            msgs = state_data.get("messages", [])
            if msgs:
                self._messages = msgs
            mem_data = state_data.get("memory", {})
            self.memory_module.from_dict(mem_data)
            logger.info(f"[Glados] State loaded from {filename}.")
        except FileNotFoundError:
            logger.warning(f"[Glados] No saved state found at {filename}, starting fresh.")
        except Exception as e:
            logger.error(f"[Glados] Error loading state: {e}")

    # -------------------------------------------------------------------------
    # +++ Audio Loop +++
    # -------------------------------------------------------------------------
    def start_listen_event_loop(self):
        self.load_state()
        self.input_stream.start()
        logger.success("[Glados] Audio Modules Operational")
        logger.success("[Glados] Listening... Press Ctrl+C to stop.")

        try:
            while True:
                sample, vad_confidence = self._sample_queue.get()
                self._handle_audio_sample(sample, vad_confidence)
        except KeyboardInterrupt:
            self.shutdown_event.set()
            self.input_stream.stop()
            self.save_state()

    async def start_listen_event_loop(self):
        self.load_state()
        self.input_stream.start()
        logger.success("[Glados] Audio Modules Operational")
        logger.success("[Glados] Listening... (async) Press Ctrl+C to stop.")

        cli_thread = threading.Thread(target=self.cli_interface)
        cli_thread.start()

        try:
            while not self.shutdown_event.is_set():
                try:
                    sample, vad_confidence = self._sample_queue.get(timeout=0.1)
                    self._handle_audio_sample(sample, vad_confidence)
                except queue.Empty:
                    await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            self.shutdown_event.set()
        finally:
            self.input_stream.stop()
            cli_thread.join()
            self.save_state()

    def _handle_audio_sample(self, sample: np.ndarray, vad_confidence: bool):
        if not self._recording_started:
            self._manage_pre_activation_buffer(sample, vad_confidence)
        else:
            self._process_activated_audio(sample, vad_confidence)

    def _manage_pre_activation_buffer(self, sample: np.ndarray, vad_confidence: bool):
        if self._buffer.full():
            self._buffer.get()
        self._buffer.put(sample)
        if vad_confidence:
            sd.stop()
            self.processing = False
            self._samples = list(self._buffer.queue)
            self._recording_started = True

    def _process_activated_audio(self, sample: np.ndarray, vad_confidence: bool):
        self._samples.append(sample)
        if not vad_confidence:
            self._gap_counter += 1
            if self._gap_counter >= PAUSE_LIMIT // VAD_SIZE:
                self._process_detected_audio()
        else:
            self._gap_counter = 0

    # -------------------------------------------------------------------------
    # +++ ASR +++
    # -------------------------------------------------------------------------
    def _process_detected_audio(self):
        logger.debug("[Glados] Detected pause, processing final chunk.")
        self.input_stream.stop()

        text = self.asr(self._samples)
        if text:
            logger.success(f"[Glados ASR] Detected text: '{text}'")
            self.last_user_input_time = time.time()
            self.memory_module.add_memory(f"UserSaid: {text}")
            self.save_state()

            if self.wake_word and not self._wakeword_detected(text):
                logger.info(f"[Glados] Wake word '{self.wake_word}' not matched.")
            else:
                self.llm_queue.put(text)
                self.processing = True
                self.currently_speaking = True

        if not self.interruptible:
            while self.currently_speaking:
                time.sleep(PAUSE_TIME)

        self.reset()
        self.input_stream.start()

    def asr(self, samples: List[np.ndarray]) -> str:
        audio = np.concatenate(samples)
        return self._asr_model.transcribe(audio)

    def reset(self):
        logger.debug("[Glados] Resetting recorder.")
        self._recording_started = False
        self._samples.clear()
        self._gap_counter = 0
        with self._buffer.mutex:
            self._buffer.queue.clear()

    def _wakeword_detected(self, text: str) -> bool:
        if not self.wake_word:
            return False
        words = text.split()
        closest_distance = min(distance(w.lower(), self.wake_word) for w in words)
        return closest_distance < SIMILARITY_THRESHOLD

    # -------------------------------------------------------------------------
    # +++ CLI +++
    # -------------------------------------------------------------------------
    def cli_interface(self):
        while not self.shutdown_event.is_set():
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                self.shutdown_event.set()
                break
            self.last_user_input_time = time.time()
            self.memory_module.add_memory(f"UserSaid: {user_input}")
            self.save_state()

            self.llm_queue.put(user_input)
            self.processing = True
            self.currently_speaking = True

            if not self.interruptible:
                while self.currently_speaking and not self.shutdown_event.is_set():
                    time.sleep(PAUSE_TIME)

            if self.messages:
                print(f"AI: {self.messages[-1]['content']}")

    # -------------------------------------------------------------------------
    # +++ TTS +++
    # -------------------------------------------------------------------------
    def process_TTS_thread(self):
        assistant_text_chunks = []
        finished = False
        interrupted = False

        while not self.shutdown_event.is_set():
            try:
                generated_text = self.tts_queue.get(timeout=PAUSE_TIME)
                if generated_text == "<EOS>":
                    finished = True
                elif not generated_text:
                    logger.warning("[TTS] Empty string, skipping.")
                else:
                    logger.success(f"[TTS] {generated_text}")

                    # If not interruptible => stop mic to avoid hearing ourselves
                    if not self.interruptible:
                        try:
                            self.input_stream.stop()
                        except Exception as e:
                            logger.warning(f"Could not stop input stream: {e}")

                    # chunk + sanitize
                    partial_chunks = chunk_text(generated_text, CHUNK_SIZE)
                    for chunk in partial_chunks:
                        audio = self._tts.generate_speech_audio(chunk)
                        total_samples = len(audio)
                        if total_samples:
                            sd.play(audio, self._tts.rate)
                            if self.interruptible:
                                tts_start_time = time.time()
                                interrupted, pct_played = self.percentage_played(total_samples, tts_start_time)
                                if interrupted:
                                    partial_txt = self.clip_interrupted_sentence(chunk, pct_played)
                                    logger.info(f"[TTS] Interrupted at {pct_played}% -> {partial_txt}")
                                    finished = True
                                    break
                            else:
                                sd.wait()
                        assistant_text_chunks.append(chunk)

                    if not self.interruptible:
                        try:
                            self.input_stream.start()
                        except Exception as e:
                            logger.warning(f"Could not restart input stream: {e}")

                if finished:
                    combined_text = " ".join(assistant_text_chunks)
                    self.messages.append({"role": "assistant", "content": combined_text})
                    self.memory_module.add_memory(f"AssistantSaid: {combined_text}")
                    self.save_state()

                    assistant_text_chunks = []
                    finished = False
                    interrupted = False
                    self.currently_speaking = False
            except queue.Empty:
                pass

    def clip_interrupted_sentence(self, text: str, pct: float) -> str:
        tokens = text.split()
        count = round((pct / 100) * len(tokens))
        partial = " ".join(tokens[:count])
        if count < len(tokens):
            partial += "<INTERRUPTED>"
        return partial

    def percentage_played(self, total_samples: int, tts_start_time: float) -> Tuple[bool, int]:
        interrupted = False
        start_time = time.time()
        while sd.get_stream().active:
            time.sleep(PAUSE_TIME)
            if self.last_user_input_time > tts_start_time:
                sd.stop()
                self.tts_queue = queue.Queue()
                interrupted = True
                break
        elapsed = time.time() - start_time + 0.12
        played_samples = elapsed * self._tts.rate
        pct = min(int((played_samples / total_samples) * 100), 100)
        return interrupted, pct

    # -------------------------------------------------------------------------
    # +++ LLM +++
    # -------------------------------------------------------------------------
    def process_LLM(self):
        while not self.shutdown_event.is_set():
            try:
                user_text = self.llm_queue.get(timeout=0.1)
                relevant = self.memory_module.get_relevant_memories(user_text, k=3)
                if relevant:
                    rag_context = "\n".join([f"- {r}" for r in relevant])
                    self._messages.append({"role": "system", "content": f"Relevant Past Context:\n{rag_context}"})

                self._messages.append({"role": "user", "content": user_text})
                prompt = self.template.render(
                    messages=self._messages,
                    bos_token="<|begin_of_text|>",
                    add_generation_prompt=True,
                )
                logger.debug("[LLM] Sending request to LLM server...")
                req_data = {"stream": True, "prompt": prompt}

                with requests.post(
                    self.completion_url, headers=self.prompt_headers, json=req_data, stream=True
                ) as response:
                    chunk_buf = []
                    for line in response.iter_lines():
                        if self.shutdown_event.is_set():
                            break
                        if self.last_user_input_time > time.time():
                            break
                        if line:
                            # decode ignoring invalid
                            decoded = line.decode("utf-8", errors="ignore").removeprefix("data: ")
                            parsed = json.loads(decoded)
                            if not parsed["stop"]:
                                token = parsed["content"]
                                chunk_buf.append(token)
                                if token in [".", "!", "?", ":", ";", "?!", "\n", "\n\n"]:
                                    self._process_sentence(chunk_buf)
                                    chunk_buf = []
                    if chunk_buf:
                        self._process_sentence(chunk_buf)
                    self.tts_queue.put("<EOS>")

            except queue.Empty:
                time.sleep(PAUSE_TIME)

    def _process_sentence(self, tokens: List[str]):
        text = "".join(tokens)
        text = re.sub(r"\*.*?\*|\(.*?\)", "", text)
        text = (
            text.replace("\n\n", ". ")
            .replace("\n", ". ")
            .replace("  ", " ")
            .replace(":", " ")
        )
        if text.strip():
            self.tts_queue.put(text)


async def start() -> None:
    llama_server_config = LlamaServerConfig.from_yaml("glados_config.yml")
    llama_server = None
    if llama_server_config is not None:
        llama_server = LlamaServer.from_config(llama_server_config)
        llama_server.start()

    glados_config = GladosConfig.from_yaml("glados_config.yml")
    if llama_server is not None:
        if glados_config.completion_url:
            raise ValueError(
                "Cannot have both a local LlamaServer and a remote completion_url!\n"
                f"Found {glados_config.completion_url=}"
            )
        glados_config.completion_url = llama_server.completion_url
    else:
        if not glados_config.completion_url:
            raise ValueError("No completion_url provided and no local LlamaServer configured.")

    glados = Glados.from_config(glados_config)
    await glados.start_listen_event_loop()

if __name__ == "__main__":
    asyncio.run(start())
