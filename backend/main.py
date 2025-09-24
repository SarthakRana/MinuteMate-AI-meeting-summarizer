import os, json, datetime
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# -------- Provider selection --------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()

# OpenAI
from openai import OpenAI as _OpenAIClient
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Gemini (Google AI Studio)
import google.generativeai as genai
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Decide actual provider
def _choose_provider() -> str:
    if LLM_PROVIDER in ("openai", "gemini"):
        return LLM_PROVIDER
    return "gemini" if GEMINI_API_KEY else "openai"

PROVIDER = _choose_provider()

# ---------- FastAPI ----------
app = FastAPI(title="Minute Mate - Your Meeting Summarizer", version="1.1.0")

# ---------- Schemas ----------
class AnalyzeIn(BaseModel):
    title: str = "Untitled Meeting"
    transcript: str
    attendees: List[str] = Field(default_factory=list)
    meeting_date: str = Field(default_factory=lambda: datetime.date.today().isoformat())

class SummaryOut(BaseModel):
    bullets: List[str] = []
    action_items: List[str] = []
    risks: List[str] = []

class SentimentOut(BaseModel):
    by_speaker: Dict[str, float] = {}
    overall: float = 0.0

class AnalyzeOut(BaseModel):
    title: str
    summary: SummaryOut
    sentiment: SentimentOut

# ---------- Utils ----------
def _coerce_json(txt: str, fallback: Any):
    if txt is None:
        return fallback
    # strip markdown code fences if present
    t = txt.strip()
    if t.startswith("```"):
        t = t.strip("`")
        # remove possible language tag
        t = t.split("\n", 1)[1] if "\n" in t else t
    try:
        return json.loads(t)
    except Exception:
        s, e = t.find("{"), t.rfind("}")
        if s != -1 and e != -1 and e > s:
            try: return json.loads(t[s:e+1])
            except: pass
        s, e = t.find("["), t.rfind("]")
        if s != -1 and e != -1 and e > s:
            try: return json.loads(t[s:e+1])
            except: pass
    return fallback

def split_by_speaker(transcript: str) -> List[Tuple[str,str]]:
    rows = []
    current_speaker = "Unknown"
    buf: List[str] = []
    for line in transcript.splitlines():
        if ":" in line:
            head, rest = line.split(":", 1)
            if len(head.split()) <= 4 and head.strip():
                if buf:
                    rows.append((current_speaker, "\n".join(buf).strip()))
                    buf = []
                current_speaker = head.strip()
                if rest.strip():
                    buf.append(rest.strip())
                continue
        if line.strip():
            buf.append(line.strip())
    if buf:
        rows.append((current_speaker, "\n".join(buf).strip()))
    return rows

def chunk_text(text: str, max_chars: int = 6000) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    parts, cur, cur_len = [], [], 0
    for para in text.split("\n"):
        if cur_len + len(para) + 1 > max_chars:
            parts.append("\n".join(cur))
            cur, cur_len = [para], len(para) + 1
        else:
            cur.append(para)
            cur_len += len(para) + 1
    if cur:
        parts.append("\n".join(cur))
    return parts

# ---------- Provider clients ----------
_openai_client = None
_gemini_model = None

def _get_openai():
    global _openai_client
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    if _openai_client is None:
        _openai_client = _OpenAIClient(api_key=OPENAI_API_KEY)
    return _openai_client

def _get_gemini():
    global _gemini_model
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    if _gemini_model is None:
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    return _gemini_model

# ---------- LLM wrappers (JSON) ----------
def llm_json(prompt: str, system: str = "Return ONLY valid JSON.") -> Any:
    """
    Sends a prompt to the chosen provider and expects JSON-only in response.
    """
    try:
        if PROVIDER == "openai":
            client = _get_openai()
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":system},
                          {"role":"user","content":prompt}],
                temperature=0.2,
            )
            content = resp.choices[0].message.content
            return _coerce_json(content, fallback=None)

        elif PROVIDER == "gemini":
            model = _get_gemini()
            # Gemini likes explicit JSON instruction
            sys = system + " Output must be raw JSON, no markdown fences."
            res = model.generate_content([sys, prompt])
            # Gemini returns a response object w/ text
            content = res.text if hasattr(res, "text") else None
            return _coerce_json(content, fallback=None)

        else:
            raise RuntimeError(f"Unsupported provider: {PROVIDER}")

    except Exception as e:
        # Standardize error
        raise HTTPException(status_code=503, detail=f"LLM error ({PROVIDER}): {e}")

# ---------- Summarization + Rollup ----------
def summarize_transcript(transcript: str) -> Dict[str, Any]:
    """
    Chunk long transcripts -> partial JSON -> roll up to final JSON.
    """
    chunks = chunk_text(transcript, max_chars=6000)
    chunk_prompt = """You are an expert meeting analyst.
Summarize the following meeting CHUNK. Return strict JSON with keys:
- "bullets": 3-5 concise bullets
- "action_items": 2-5 short imperative tasks
- "risks": 0-3 short risks/open questions

Chunk:
---
{chunk}
---"""

    partials: List[Dict[str, Any]] = []
    for ch in chunks:
        data = llm_json(
            prompt=chunk_prompt.format(chunk=ch),
            system="Return ONLY valid JSON with keys bullets, action_items, risks."
        )
        if not isinstance(data, dict):
            data = {"bullets": [], "action_items": [], "risks": []}
        data.setdefault("bullets", [])
        data.setdefault("action_items", [])
        data.setdefault("risks", [])
        partials.append(data)

    rollup_prompt = f"""Merge multiple partial meeting summaries into a single overall summary.
Input is a JSON array of objects with keys "bullets", "action_items", "risks".
Return strict JSON with:
- "bullets": 5-8 concise bullets
- "action_items": 3-8 short, deduplicated tasks
- "risks": up to 3 major risks

Partials:
{json.dumps(partials, ensure_ascii=False)}"""

    merged = llm_json(
        prompt=rollup_prompt,
        system="Return ONLY valid JSON with keys bullets, action_items, risks."
    )
    if not isinstance(merged, dict):
        merged = {"bullets": [], "action_items": [], "risks": []}
    merged.setdefault("bullets", [])
    merged.setdefault("action_items", [])
    merged.setdefault("risks", [])
    return merged

# ---------- Sentiment ----------
def sentiment_by_speaker(turns: List[Tuple[str,str]]) -> Dict[str, float]:
    """
    Ask the model to give sentiment per speaker in {-1,0,1} averaged over their turns.
    """
    if not turns:
        return {}

    # Limit each utterance to keep prompts small
    items = [{"speaker": s, "text": (t[:1000] if t else "")} for s, t in turns]
    prompt = f"""For each speaker's texts, classify sentiment as -1 (negative), 0 (neutral), or 1 (positive).
Aggregate multiple turns by the same speaker by averaging. 
Return a JSON object mapping speaker -> average sentiment (float rounded to 2 decimals).

Items (JSON array):
{json.dumps(items, ensure_ascii=False)}"""

    out = llm_json(
        prompt=prompt,
        system='Return ONLY a JSON object like {"Alice": 0.33, "Bob": -0.5}. No extra text.'
    )
    if not isinstance(out, dict):
        return {}
    by_speaker: Dict[str, float] = {}
    for k, v in out.items():
        try:
            by_speaker[k] = float(v)
        except Exception:
            continue
    return by_speaker

# ---------- Routes ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": PROVIDER,
        "openai_model": OPENAI_MODEL if PROVIDER == "openai" else None,
        "gemini_model": GEMINI_MODEL if PROVIDER == "gemini" else None,
        "project": OPENAI_PROJECT or None
    }

@app.post("/analyze", response_model=AnalyzeOut)
def analyze(payload: AnalyzeIn):
    if not payload.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty")

    try:
        summary = summarize_transcript(payload.transcript)
        turns = split_by_speaker(payload.transcript)
        by_speaker = sentiment_by_speaker(turns)
        overall = round(sum(by_speaker.values())/len(by_speaker), 2) if by_speaker else 0.0

        return AnalyzeOut(
            title=payload.title,
            summary=SummaryOut(
                bullets=summary.get("bullets", []),
                action_items=summary.get("action_items", []),
                risks=summary.get("risks", [])
            ),
            sentiment=SentimentOut(
                by_speaker=by_speaker,
                overall=overall
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {e}")