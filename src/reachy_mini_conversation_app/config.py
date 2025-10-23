import os
import logging
from typing import Any, Dict
from pathlib import Path
from typing import Any, Dict

import yaml  # type: ignore[import-untyped]
from dotenv import load_dotenv


logger = logging.getLogger(__name__)

# Check if .env file exists
env_file = Path(".env")
if not env_file.exists():
    raise RuntimeError(
        ".env file not found. Please create one based on .env.example:\n"
        "  cp .env.example .env\n"
        "Then add your OPENAI_API_KEY to the .env file.",
    )

# Load .env and verify it was loaded successfully
if not load_dotenv():
    raise RuntimeError(
        "Failed to load .env file. Please ensure the file is readable and properly formatted.",
    )

logger.info("Configuration loaded from .env file")



def _load_cascade_config() -> Dict[str, Any]:
    """Load cascade configuration from YAML file."""
    config_file = Path("cascade.yaml")

    if not config_file.exists():
        raise RuntimeError(
            "cascade.yaml not found. Please create it:\n"
            "The cascade.yaml file configures ASR, LLM, and TTS providers."
        )

    try:
        with open(config_file) as f:
            config: Dict[str, Any] = yaml.safe_load(f)

        # Validate required top-level keys
        required_keys = ['asr', 'llm', 'tts']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise RuntimeError(
                f"cascade.yaml is missing required keys: {', '.join(missing_keys)}\n"
            )

        logger.info("Cascade configuration loaded from cascade.yaml")
        return config

    except yaml.YAMLError as e:
        raise RuntimeError(
            f"Invalid YAML syntax in cascade.yaml:\n{e}\n"
            "Please check the file for syntax errors."
        )




class Config:
    """Configuration class for the conversation app."""

    # Required
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not set in .env file. Please add it:\n"
            "  OPENAI_API_KEY=your_api_key_here",
        )
    if not OPENAI_API_KEY.strip():
        raise RuntimeError(
            "OPENAI_API_KEY is empty in .env file. Please provide a valid API key.",
        )

    # Optional
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-realtime")
    HF_HOME = os.getenv("HF_HOME", "./cache")
    LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Optional, falls back to hf auth login if not set

    logger.debug(f"Model: {MODEL_NAME}, HF_HOME: {HF_HOME}, Vision Model: {LOCAL_VISION_MODEL}")

    # ---------------------
    # Cascade configuration
    # ---------------------

    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    _cascade = _load_cascade_config()

    # ASR configuration
    CASCADE_ASR_PROVIDER = _cascade['asr']['provider']
    PARAKEET_MODEL = _cascade['asr']['parakeet']['model']
    PARAKEET_PRECISION = _cascade['asr']['parakeet']['precision']

    # LLM configuration
    CASCADE_LLM_PROVIDER = _cascade['llm']['provider']
    CASCADE_LLM_MODEL = _cascade['llm']['openai_gpt']['model']
    GEMINI_MODEL = _cascade['llm']['gemini']['model']

    # TTS configuration
    CASCADE_TTS_PROVIDER = _cascade['tts']['provider']
    CASCADE_TTS_TRIM_SILENCE = _cascade['tts']['trim_silence']
    CASCADE_TTS_VOICE = _cascade['tts']['openai_tts']['voice']
    # Store all TTS provider settings for easy access
    _tts_openai = _cascade['tts']['openai_tts']
    _tts_kokoro = _cascade['tts']['kokoro']
    _tts_elevenlabs = _cascade['tts']['elevenlabs']

    # Provider-specific TTS settings
    KOKORO_VOICE = _tts_kokoro['voice']
    ELEVENLABS_VOICE_ID = _tts_elevenlabs['voice_id']
    ELEVENLABS_MODEL = _tts_elevenlabs['model']

    logger.debug(
        f"Cascade: ASR={CASCADE_ASR_PROVIDER}, LLM={CASCADE_LLM_PROVIDER} ({CASCADE_LLM_MODEL}), "
        f"TTS={CASCADE_TTS_PROVIDER} (trim_silence={CASCADE_TTS_TRIM_SILENCE})"
    )
    if CASCADE_ASR_PROVIDER == "parakeet":
        logger.debug(f"Parakeet: model={PARAKEET_MODEL}, precision={PARAKEET_PRECISION}")
    if CASCADE_LLM_PROVIDER == "gemini":
        logger.debug(f"Gemini: model={GEMINI_MODEL}")
    if CASCADE_TTS_PROVIDER == "openai_tts":
        logger.debug(f"OpenAI TTS: voice={CASCADE_TTS_VOICE}")
    elif CASCADE_TTS_PROVIDER == "kokoro":
        logger.debug(f"Kokoro: voice={KOKORO_VOICE}")
    elif CASCADE_TTS_PROVIDER == "elevenlabs":
        logger.debug(f"ElevenLabs: voice_id={ELEVENLABS_VOICE_ID}, model={ELEVENLABS_MODEL}")


config = Config()
