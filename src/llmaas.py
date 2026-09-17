"""Step 2 - talking to the LLMaaS gateway. The only file here that goes online.

Run it on its own to check the credentials and see what the model says:

    python src/llmaas.py                      # just a connection check
    python src/llmaas.py photo.jpg            # send a photo, print the JSON
    python src/llmaas.py front.jpg back.jpg   # both faces, one call
"""

import base64
import io
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

IDP_URL = "https://idp.cloud.vwgroup.com/auth/realms/kums-mfa/protocol/openid-connect/token"
BASE_URL = "https://llmapi.ai.vwgroup.com"

# Phone photos come in around 4000px and cost a fortune in tokens for detail
# nobody uses. 1024 is about where label text is still readable, which is the
# hardest thing we ask of the model.
MAX_IMAGE_PX = 1024

_token = None
_tokenExpires = 0.0


class Unavailable(RuntimeError):
    """Gateway can't be reached, or we're not set up to reach it."""


def _loadEnv():
    """Read the .env next to the project into os.environ."""
    envFile = Path(__file__).resolve().parent.parent / ".env"
    if not envFile.exists():
        logger.debug("no .env at %s, using the real environment", envFile)
        return

    loaded = 0
    for line in envFile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # People quote things out of shell habit; strip them or the quotes end
        # up inside the secret.
        value = value.strip().strip('"').strip("'")
        # Anything already in the real environment wins, so you can still do
        # LLMAAS_API_KEY=... python src/server.py for a one-off.
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    logger.debug("read %d setting(s) from %s", loaded, envFile.name)


_loadEnv()

CLIENT_ID = os.getenv("LLMAAS_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("LLMAAS_CLIENT_SECRET", "")
API_KEY = os.getenv("LLMAAS_API_KEY", "")          # the sk-... one
MODEL = os.getenv("LLMAAS_VISION_MODEL", "gpt-4o")


def isConfigured():
    """(ok, reason) - so the page can say what's missing instead of just failing.

    There are two separate credentials and people miss the second one. Without
    it the gateway answers 401 "Malformed API Key passed in", which sounds like
    the token you DO have is broken. It isn't - the other header is just absent.
    """
    if not CLIENT_ID or not CLIENT_SECRET:
        return False, "LLMAAS_CLIENT_ID / LLMAAS_CLIENT_SECRET are not set."
    if not API_KEY:
        return False, ("LLMAAS_API_KEY is not set. It's the LiteLLM virtual key "
                       "starting with 'sk-', separate from the client id/secret.")
    if not API_KEY.startswith("sk-"):
        return False, f"LLMAAS_API_KEY should start with 'sk-', got {API_KEY[:6]!r}."
    return True, "ready"


def getToken():
    """A live IDP access token, cached until it's nearly expired."""
    global _token, _tokenExpires
    if _token and time.time() < _tokenExpires:
        logger.debug("reusing cached token, %ds left", int(_tokenExpires - time.time()))
        return _token

    import httpx
    logger.debug("asking the IDP for a token")
    try:
        response = httpx.post(IDP_URL, timeout=30, data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "client_credentials",
        })
    except Exception as error:
        logger.error("IDP unreachable: %s", error)
        raise Unavailable(f"Cloud IDP unreachable: {error}")

    if response.status_code != 200:
        logger.error("IDP rejected the credentials (%s): %s",
                     response.status_code, response.text[:200])
        raise Unavailable(f"Cloud IDP rejected the credentials "
                          f"({response.status_code}): {response.text[:200]}")

    payload = response.json()
    _token = payload["access_token"]
    # Refresh a minute early so a request never starts on a token that dies
    # halfway through.
    _tokenExpires = time.time() + int(payload.get("expires_in", 300)) - 60
    logger.info("token acquired, good for %ds", int(_tokenExpires - time.time()))
    return _token


def getClient():
    """An OpenAI client pointed at the gateway, with both auth headers on it."""
    ok, reason = isConfigured()
    if not ok:
        logger.warning("not configured: %s", reason)
        raise Unavailable(reason)

    from openai import OpenAI
    return OpenAI(
        api_key=getToken(),
        base_url=BASE_URL,
        default_headers={"X-LLM-API-CLIENT-ID": f"Bearer {API_KEY}"},
        timeout=90,
        max_retries=2,
    )


def _encodeImage(blob):
    """Shrink to MAX_IMAGE_PX and base64 it."""
    try:
        from PIL import Image
    except ImportError:
        # No Pillow, fine - send it full size. Costs tokens, still works.
        logger.warning("Pillow isn't installed, sending the image unresized")
        return base64.b64encode(blob).decode()

    try:
        image = Image.open(io.BytesIO(blob))
        original = image.size
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        if max(image.size) > MAX_IMAGE_PX:
            scale = MAX_IMAGE_PX / max(image.size)
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        logger.debug("image %dx%d -> %dx%d, %d KB",
                     original[0], original[1], image.width, image.height,
                     buffer.tell() // 1024)
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as error:
        # Broken or exotic image - let the gateway have a go at the original.
        logger.warning("couldn't resize (%s), sending the original", error)
        return base64.b64encode(blob).decode()


def look(images, systemPrompt, schema, userMessage):
    """Send the photos, get the raw dict back.

    All the images go in ONE call. Three photos of the same crate should give
    one opinion, not three that then need arguing about - and "worst damage on
    any face" is a call you can only make with all the faces in front of you.
    """
    if not images:
        raise ValueError("look() needs at least one image")

    content = [{"type": "text", "text": userMessage}]
    for blob in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_encodeImage(blob)}",
                          "detail": "high"},
        })

    logger.info("sending %d image(s) to %s", len(images), MODEL)
    started = time.time()

    completion = getClient().chat.completions.create(
        model=MODEL,
        temperature=0,                       # inspection, not creative writing
        messages=[{"role": "system", "content": systemPrompt},
                  {"role": "user", "content": content}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "observation",
                                         "schema": schema,
                                         "strict": True}},
    )

    raw = completion.choices[0].message.content or "{}"
    try:
        reply = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("gateway returned non-JSON: %s", raw[:300])
        raise Unavailable(f"Gateway returned something that isn't JSON: {raw[:200]}")

    logger.info("model answered in %.1fs: %s / %s / %s / label %s",
                time.time() - started,
                reply.get("object_category"), reply.get("damage_type"),
                reply.get("damage_severity"), reply.get("label_state"))
    return reply


def main(argv):
    """Run this file on its own: images in, the model's raw JSON out."""
    import prompt

    ok, reason = isConfigured()
    print(f"gateway   {BASE_URL}")
    print(f"model     {MODEL}")
    print(f"config    {'ok' if ok else 'NOT READY - ' + reason}")
    if not ok:
        return 1

    try:
        getToken()
    except Unavailable as error:
        print(f"token     FAILED - {error}")
        return 1

    paths = argv[1:]
    if not paths:
        print("\nNo images given, so nothing was sent.")
        print("Try: python src/llmaas.py photo.jpg")
        return 0

    images = []
    for path in paths:
        blob = Path(path).read_bytes()
        images.append(blob)
        print(f"image     {path}  ({len(blob) // 1024} KB)")

    systemPrompt, schema = prompt.build()
    print()
    reply = look(images, systemPrompt, schema, prompt.userText(len(images)))
    print()
    print(json.dumps(reply, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s  %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S")
    # httpcore logs every HTTP frame at DEBUG and buries everything of ours.
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    sys.exit(main(sys.argv))
