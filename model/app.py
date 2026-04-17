from __future__ import annotations
import os
import json
import time
import logging
import pickle
from pathlib import Path
from typing import Optional, List

import torch
import tiktoken
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from model.gpt import GPT, GPTConfig, load_config_from_meta
    from model.test_generator import TestGeneratorRequest, dispatch
except ModuleNotFoundError:
    from gpt import GPT, GPTConfig, load_config_from_meta
    from test_generator import TestGeneratorRequest, dispatch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("seccodegpt.api")


CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "data/seccode/ckpt.pt")
META_PATH       = os.getenv("META_PATH",       "data/seccode/meta.pkl")
DEVICE_ENV      = os.getenv("DEVICE",          "")

def _resolve_device() -> str:
    if DEVICE_ENV:
        return DEVICE_ENV
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _resolve_device()

_model: GPT | None = None
_enc:   tiktoken.Encoding | None = None
_vocab_size: int = 0


def _load_model() -> tuple[GPT, tiktoken.Encoding]:
    global _model, _enc, _vocab_size

    if _model is not None and _enc is not None:
        return _model, _enc

    log.info("Loading BPE encoding …")
    _enc = tiktoken.get_encoding("cl100k_base")

    cfg = load_config_from_meta(META_PATH)

    ckpt_path = Path(CHECKPOINT_PATH)
    if ckpt_path.exists():
        log.info(f"Loading checkpoint from {ckpt_path} …")
        checkpoint = torch.load(str(ckpt_path), map_location=DEVICE)

        # Support checkpoints that embed the config
        if "model_args" in checkpoint:
            for k, v in checkpoint["model_args"].items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

        _model = GPT(cfg)
        state  = checkpoint.get("model", checkpoint)
        # Strip DDP prefix if present
        state  = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
        _model.load_state_dict(state, strict=False)
        log.info("Checkpoint loaded successfully.")
    else:
        log.warning(
            f"Checkpoint not found at {ckpt_path}. "
            "Initialising untrained model — outputs will be random."
        )
        _model = GPT(cfg)

    _model.eval()
    _model.to(DEVICE)
    _vocab_size = cfg.vocab_size
    return _model, _enc

app = FastAPI(
    title="SecCodeGPT API",
    description=(
        "REST interface for the SecCodeGPT security automation LLM. "
        "Generates YAML templates, JSON payloads, pytest suites, and Robot Framework tests."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    """Pre-warm the model at startup so the first request is not slow."""
    try:
        _load_model()
        log.info(f"SecCodeGPT model ready — device={DEVICE}, vocab_size={_vocab_size}")
    except Exception as exc:
        log.error(f"Model load failed: {exc}")

class GenerateRequest(BaseModel):
    prompt:         str   = Field(..., description="Input prompt for the LLM.")
    max_new_tokens: int   = Field(256,  ge=1,   le=2048, description="Max tokens to generate.")
    temperature:    float = Field(0.8,  ge=0.1, le=2.0,  description="Sampling temperature.")
    top_k:          int   = Field(200,  ge=1,   le=1000, description="Top-k filtering.")


class GenerateResponse(BaseModel):
    prompt:     str
    output:     str
    n_tokens:   int
    elapsed_ms: float


class TestGenerateRequest(BaseModel):
    tool:               str            = Field(...,  description="Security tool: nmap | nuclei | sqlmap | nikto | curl")
    target:             str            = Field(...,  description="Target host or URL.")
    framework:          str            = Field("pytest", description="Test framework: pytest | robot")
    timeout:            int            = Field(60,   ge=5, le=300, description="Command timeout in seconds.")
    assert_json:        bool           = Field(False, description="Assert output is valid JSON (pytest only).")
    expected_keywords:  List[str]      = Field([],   description="Keywords that must appear in the output.")
    extra_args:         List[str]      = Field([],   description="Extra CLI args appended to the command.")


class TestGenerateResponse(BaseModel):
    framework:  str
    filepath:   str
    content:    str
    command:    List[str]
    elapsed_ms: float


class HealthResponse(BaseModel):
    status:     str
    device:     str
    vocab_size: int
    checkpoint: str

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    """Service health check."""
    return HealthResponse(
        status="ok" if _model is not None else "model_not_loaded",
        device=DEVICE,
        vocab_size=_vocab_size,
        checkpoint=CHECKPOINT_PATH,
    )


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
def generate_text(req: GenerateRequest):
    """
    Raw text generation from a free-form prompt.
    """
    model, enc = _load_model()

    t0 = time.time()
    try:
        tokens = enc.encode(req.prompt)
        idx    = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)
        out    = model.generate(idx, req.max_new_tokens, req.temperature, req.top_k)
        output = enc.decode(out[0].tolist())
    except Exception as exc:
        log.exception("Generation error")
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = (time.time() - t0) * 1000
    return GenerateResponse(
        prompt=req.prompt,
        output=output,
        n_tokens=len(out[0]) - len(tokens),
        elapsed_ms=round(elapsed, 2),
    )


@app.post("/generate-yaml", response_model=GenerateResponse, tags=["Generation"])
def generate_yaml(req: GenerateRequest):
    """
    Generate a Nuclei-style YAML security template.
    The model output is wrapped with a structured YAML prefix.
    """
    model, enc = _load_model()

    t0 = time.time()
    try:
        output = model.generate_yaml(req.prompt, enc, req.max_new_tokens, DEVICE)
    except Exception as exc:
        log.exception("YAML generation error")
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = (time.time() - t0) * 1000
    return GenerateResponse(
        prompt=req.prompt,
        output=output,
        n_tokens=len(enc.encode(output)),
        elapsed_ms=round(elapsed, 2),
    )


@app.post("/generate-json", response_model=GenerateResponse, tags=["Generation"])
def generate_json_endpoint(req: GenerateRequest):
    """
    Generate a structured JSON security payload.
    """
    model, enc = _load_model()

    t0 = time.time()
    try:
        output = model.generate_json(req.prompt, enc, req.max_new_tokens, DEVICE)
    except Exception as exc:
        log.exception("JSON generation error")
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = (time.time() - t0) * 1000
    return GenerateResponse(
        prompt=req.prompt,
        output=output,
        n_tokens=len(enc.encode(output)),
        elapsed_ms=round(elapsed, 2),
    )


@app.post("/generate-test", response_model=TestGenerateResponse, tags=["Test Generation"])
def generate_test(req: TestGenerateRequest):
    """
    Core endpoint — generates a fully executable test file.

    Accepts a JSON payload describing a security scan, returns a pytest
    or Robot Framework file ready to run via `pytest` / `robot`.

    Example payload:
    ```json
    {
        "tool": "nmap",
        "target": "scanme.nmap.org",
        "framework": "pytest",
        "timeout": 60,
        "expected_keywords": ["open", "80/tcp"]
    }
    ```
    """
    t0 = time.time()
    try:
        gen_req = TestGeneratorRequest(
            tool=req.tool,
            target=req.target,
            framework=req.framework,
            timeout=req.timeout,
            assert_json=req.assert_json,
            expected_keywords=req.expected_keywords,
            extra_args=req.extra_args,
        )
        result = dispatch(gen_req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.exception("Test generation error")
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = (time.time() - t0) * 1000
    log.info(f"/generate-test [{req.framework}] tool={req.tool} target={req.target} → {result['filepath']}")

    return TestGenerateResponse(
        framework=result["framework"],
        filepath=result["filepath"],
        content=result["content"],
        command=result["command"],
        elapsed_ms=round(elapsed, 2),
    )