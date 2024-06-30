import importlib.metadata
import os
import requests

from io import BytesIO
from os import path
from typing import Annotated, BinaryIO, Union
from urllib.parse import quote


import ffmpeg
import numpy as np
# from fastapi import FastAPI, File, Query, UploadFile, applications
from fastapi import FastAPI, File, Form, Query, UploadFile, applications, Depends, HTTPException, Security
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.security import APIKeyHeader
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from whisper import tokenizer

# Get the TOKEN from environment variables
API_TOKEN = os.getenv("TOKEN")

if not API_TOKEN:
    raise ValueError("Please set the TOKEN environment variable")

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header is None or api_key_header != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key_header

ASR_ENGINE = os.getenv("ASR_ENGINE", "openai_whisper")

if ASR_ENGINE == "faster_whisper":
    from .faster_whisper.core import language_detection, transcribe
else:
    from .openai_whisper.core import language_detection, transcribe

SAMPLE_RATE = 16000
LANGUAGE_CODES = sorted(tokenizer.LANGUAGES.keys())

projectMetadata = importlib.metadata.metadata("whisper-asr-webservice")
app = FastAPI(
    title=projectMetadata["Name"].title().replace("-", " "),
    description=projectMetadata["Summary"],
    version=projectMetadata["Version"],
    contact={"url": projectMetadata["Home-page"]},
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    license_info={"name": "MIT License", "url": projectMetadata["License"]},
)

assets_path = os.getcwd() + "/swagger-ui-assets"
if path.exists(assets_path + "/swagger-ui.css") and path.exists(assets_path + "/swagger-ui-bundle.js"):
    app.mount("/assets", StaticFiles(directory=assets_path), name="static")

    def swagger_monkey_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_favicon_url="",
            swagger_css_url="/assets/swagger-ui.css",
            swagger_js_url="/assets/swagger-ui-bundle.js",
        )

    applications.get_swagger_ui_html = swagger_monkey_patch


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"

@app.post("/v2/audio/transcriptions", tags=["Endpoints"])
async def asr(
    api_key: str = Depends(get_api_key),
    audio_file: Union[UploadFile, None] = File(default=None),
    audio_url: Union[str, None] = Form(default=None, description="URL of the audio file to transcribe"),
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initial_prompt: Union[str, None] = Query(default=None),
    vad_filter: Annotated[
        bool | None,
        Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(True if ASR_ENGINE == "faster_whisper" else False),
        ),
    ] = False,
    word_timestamps: bool = Query(default=False, description="Word level timestamps"),
    output: Union[str, None] = Query(default="txt", enum=["txt", "vtt", "srt", "tsv", "json"]),
):
    if audio_file is None and audio_url is None:
        raise HTTPException(status_code=400, detail="Either audio_file or audio_url must be provided")

    if audio_file:
        audio_data = audio_file.file
        filename = audio_file.filename
    else:
        try:
            response = requests.get(audio_url)
            response.raise_for_status()
            audio_data = BytesIO(response.content)
            filename = audio_url.split("/")[-1]
        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to download audio file: {str(e)}")

    result = transcribe(
        load_audio(audio_data, encode), task, language, initial_prompt, vad_filter, word_timestamps, output
    )
    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            "Asr-Engine": ASR_ENGINE,
            "Content-Disposition": f'attachment; filename="{quote(filename)}.{output}"',
        },
    )

@app.post("/v1/audio/transcriptions", tags=["Endpoints"])
async def asr(
    api_key: str = Depends(get_api_key),
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initial_prompt: Union[str, None] = Query(default=None),
    vad_filter: Annotated[
        bool | None,
        Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(True if ASR_ENGINE == "faster_whisper" else False),
        ),
    ] = False,
    word_timestamps: bool = Query(default=False, description="Word level timestamps"),
    output: Union[str, None] = Query(default="txt", enum=["txt", "vtt", "srt", "tsv", "json"]),
):
    result = transcribe(
        load_audio(audio_file.file, encode), task, language, initial_prompt, vad_filter, word_timestamps, output
    )
    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            "Asr-Engine": ASR_ENGINE,
            "Content-Disposition": f'attachment; filename="{quote(audio_file.filename)}.{output}"',
        },
    )

@app.post("/detect-language", tags=["Endpoints"])
async def detect_language(
    api_key: str = Depends(get_api_key),
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through FFmpeg"),
):
    detected_lang_code = language_detection(load_audio(audio_file.file, encode))
    return {"detected_language": tokenizer.LANGUAGES[detected_lang_code], "language_code": detected_lang_code}

def load_audio(file: Union[BinaryIO, BytesIO], encode=True, sr: int = SAMPLE_RATE):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
    Parameters
    ----------
    file: Union[BinaryIO, BytesIO]
        The audio file like object or BytesIO object
    encode: Boolean
        If true, encode audio stream to WAV before sending to whisper
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    if encode:
        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input("pipe:", threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=file.read())
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    else:
        out = file.read()

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
