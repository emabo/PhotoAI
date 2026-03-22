#!/usr/bin/env python3
"""
caption_pipeline.py

Current behavior:
- Generate caption_en (BLIP-2 Flan-T5 XXL)
- Translate caption_en -> caption_it (NLLB)
- Extract tags from caption_en (English) and translate them to Italian
- Save to SQLite:
    - captions.caption = caption_it
    - captions.lang = 'it'
        - captions.params_json includes caption_en (+ useful metadata)
    - tags: save only tags in Italian (for Italian search)
- Embeddings only from caption_it (OpenCLIP ViT-L/14) -> Chroma captions collection

Requirements:
  pip install -U torch transformers accelerate sentencepiece python-dotenv pillow
  pip install -U chromadb open-clip-torch
"""

from __future__ import annotations

import os
import json
import re
import sqlite3
from pathlib import Path, PurePosixPath
from typing import Dict, Any, List, Tuple, Optional

from PIL import Image, ImageFile

import torch
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

import chromadb
from chromadb.config import Settings

import open_clip

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ------------------ env helpers ------------------

def must_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"{name} is not set (define it in .env or environment variables).")
    return v


def normalize_relpath(p: str) -> Optional[str]:
    """Normalize and validate a relative path (no absolute path, no root escape)."""
    if not p:
        return None
    p = p.strip().replace("\\", "/")
    # reject Linux/Windows absolute paths
    if p.startswith("/") or (len(p) >= 2 and p[1] == ":" and p[0].isalpha()):
        return None
    pp = PurePosixPath(p)
    parts: List[str] = []
    for part in pp.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts) if parts else None


def dir_context_from_relpath(relpath: str, levels: int = 2) -> str:
    """Last N directory levels used as human-readable context."""
    pp = PurePosixPath(relpath)
    dirs = pp.parts[:-1]
    if not dirs:
        return ""
    take = dirs[-levels:] if levels > 0 else dirs
    return " / ".join(take)


def ctx_to_keywords(dir_ctx: str, max_words: int = 10) -> str:
    """Clean folder context into a short keyword string."""
    s = (dir_ctx or "").lower()
    s = s.replace("/", " ")
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    return " ".join(s.split()[:max_words])


# ------------------ SQLite ------------------

def sqlite_connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def fetch_jobs(con: sqlite3.Connection, step: str, limit: int) -> List[str]:
    cur = con.execute(
        "SELECT sha1 FROM jobs WHERE step=? AND status='queued' LIMIT ?",
        (step, limit),
    )
    return [r[0] for r in cur.fetchall()]


def mark_job(con: sqlite3.Connection, sha1: str, step: str, status: str, detail: str = ""):
    con.execute(
        """
        INSERT INTO jobs (sha1, step, status, detail, updated_at)
        VALUES (?, ?, ?, ?, unixepoch())
        ON CONFLICT(sha1, step) DO UPDATE SET
          status=excluded.status,
          detail=excluded.detail,
          updated_at=excluded.updated_at
        """,
        (sha1, step, status, detail),
    )


def get_image_relpath(con: sqlite3.Connection, sha1: str) -> str:
    cur = con.execute("SELECT path FROM images WHERE sha1=?", (sha1,))
    row = cur.fetchone()
    if not row:
        raise RuntimeError("sha1 not found in images table")
    return row[0]


def upsert_caption_it(con: sqlite3.Connection, sha1: str, caption_it: str, caption_en: str,
                      model_name: str, params: Dict[str, Any]):
    """
    Save Italian caption as primary value (captions.caption/lang='it').
    caption_en is stored in params_json.
    """
    params = dict(params)
    params["caption_en"] = caption_en

    con.execute(
        """
        INSERT INTO captions (sha1, caption, lang, model, params_json, created_at, updated_at)
        VALUES (?, ?, 'it', ?, ?, unixepoch(), unixepoch())
        ON CONFLICT(sha1) DO UPDATE SET
          caption=excluded.caption,
          lang=excluded.lang,
          model=excluded.model,
          params_json=excluded.params_json,
          updated_at=excluded.updated_at
        """,
        (sha1, caption_it, model_name, json.dumps(params, ensure_ascii=False)),
    )


def replace_tags(con: sqlite3.Connection, sha1: str, tags_it: List[Tuple[str, float, str]]):
    """
    Save only Italian tags in the tags table.
    """
    con.execute("DELETE FROM tags WHERE sha1=?", (sha1,))
    con.executemany(
        "INSERT INTO tags (sha1, tag, score, source, created_at) VALUES (?, ?, ?, ?, unixepoch())",
        [(sha1, t, float(score), src) for (t, score, src) in tags_it],
    )


# ------------------ Tag extraction (English) ------------------

STOPWORDS_EN = set("""
a an and are as at be by for from has have in is it its of on or that the to was were with this those these
into over under near around above below between among
""".split())


def extract_tags_en(text_en: str, dir_ctx: str = "", max_tags: int = 14) -> List[Tuple[str, float, str]]:
    """
    Extract deterministic tags from an English caption:
    - tokenize, remove stopwords, filter short words and long numeric tokens
    - include keywords from directory context with slightly lower score
    Returns list (tag_en, score, source)
    """
    tags: Dict[str, Tuple[float, str]] = {}

    def add(tag: str, score: float, src: str):
        tag = tag.strip().lower()
        tag = re.sub(r"[^a-z0-9\s]+", "", tag)
        tag = re.sub(r"\s+", " ", tag).strip()
        if len(tag) < 3:
            return
        if tag in STOPWORDS_EN:
            return
        prev = tags.get(tag)
        if prev is None or score > prev[0]:
            tags[tag] = (score, src)

    # caption words
    words = re.findall(r"[a-zA-Z0-9]+", (text_en or "").lower())
    for w in words:
        if w in STOPWORDS_EN:
            continue
        if len(w) < 3:
            continue
        if w.isdigit() and len(w) >= 5:
            continue
        add(w, 1.0, "caption_en")

    # dir context keywords (often IT-ish, still useful as additional hints)
    ctx = ctx_to_keywords(dir_ctx, max_words=10)
    for w in re.findall(r"[a-zA-Z0-9]+", ctx.lower()):
        if w and w not in STOPWORDS_EN and len(w) >= 3:
            add(w, 0.7, "dirctx")

    items = sorted(tags.items(), key=lambda kv: (-kv[1][0], kv[0]))
    return [(t, sc, src) for t, (sc, src) in items[:max_tags]]


def normalize_tag_it(tag: str) -> str:
    t = (tag or "").strip().lower()
    t = re.sub(r"[^a-z0-9àèéìòóù\s]+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ------------------ Models ------------------

def make_caption_prompt_en(dir_ctx: str) -> str:
    kw = ctx_to_keywords(dir_ctx, max_words=10)
    if kw:
        return (
            f"Folder context (human notes): {kw}\n"
            "Describe what is visible in the image in ENGLISH.\n"
            "Use the folder context only if it matches what you actually see.\n"
            "Do not invent details.\n"
        )
    return (
        "Describe what is visible in the image in ENGLISH.\n"
        "Do not invent details.\n"
    )


def load_blip2(model_id: str, device: str, dtype: torch.dtype):
    processor = Blip2Processor.from_pretrained(model_id, use_fast=False, legacy=True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto" if device.startswith("cuda") else None,
    )
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    return processor, model


@torch.inference_mode()
def generate_caption_en(processor, model, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0], skip_special_tokens=True).strip()


def load_translator_nllb(model_id: str, device: str, dtype: torch.dtype):
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=dtype).to(device)
    mdl.eval()
    return tok, mdl


@torch.inference_mode()
def translate_en_to_it_nllb(tok, mdl, text: str, device: str, max_new_tokens: int = 128) -> str:
    # NLLB-200 codes
    if hasattr(tok, "src_lang"):
        tok.src_lang = "eng_Latn"
    inputs = tok(text, return_tensors="pt", truncation=True).to(device)
    forced_bos = tok.convert_tokens_to_ids("ita_Latn")
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if forced_bos is not None:
        gen_kwargs["forced_bos_token_id"] = forced_bos
    out = mdl.generate(**inputs, **gen_kwargs)
    return tok.batch_decode(out, skip_special_tokens=True)[0].strip()


@torch.inference_mode()
def translate_tags_en_to_it_nllb(tok, mdl, tags_en: List[str], device: str, max_new_tokens: int = 16) -> List[str]:
    """
    Translate a list of English tags to Italian in batch (faster than one-by-one).
    Each tag is a mini phrase; a low max_new_tokens value is enough.
    """
    if not tags_en:
        return []
    if hasattr(tok, "src_lang"):
        tok.src_lang = "eng_Latn"

    # batch tokenization
    inputs = tok(tags_en, return_tensors="pt", padding=True, truncation=True).to(device)
    forced_bos = tok.convert_tokens_to_ids("ita_Latn")
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if forced_bos is not None:
        gen_kwargs["forced_bos_token_id"] = forced_bos
    out = mdl.generate(**inputs, **gen_kwargs)
    decoded = tok.batch_decode(out, skip_special_tokens=True)
    return [d.strip() for d in decoded]


def load_openclip_text(device: str):
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", force_quick_gelu=True)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model = model.to(device).eval()
    return model, tokenizer


@torch.inference_mode()
def embed_text_openclip(model, tokenizer, device: str, text: str) -> List[float]:
    tokens = tokenizer([text]).to(device)
    feats = model.encode_text(tokens)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.detach().cpu().float().tolist()[0]


# ------------------ Chroma ------------------

def chroma_client(chroma_dir: Path):
    return chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )




