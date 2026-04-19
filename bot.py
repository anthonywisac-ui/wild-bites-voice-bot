"""
Wild Bites Voice Ordering Bot
==============================
WhatsApp Business Calling API voice bot using:
- Pipecat (orchestration)
- Deepgram (STT + TTS)
- Groq llama-3.1-8b-instant (LLM)
- SmallWebRTCTransport (Meta WhatsApp leg)

When a customer calls the Meta test number on WhatsApp,
this bot answers, takes their order via voice, and hands off
to the restaurant-bot for order confirmation + manager notification.
"""

import os
import sys
import aiohttp
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from loguru import logger
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn

# ── Pipecat imports ──────────────────────────────────────────
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection
from pipecat.transports.whatsapp.api import WhatsAppWebhookRequest
from pipecat.transports.whatsapp.client import WhatsAppClient

load_dotenv()

# ── Config ───────────────────────────────────────────────────
PORT = int(os.getenv("PORT", "8000"))
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", "")
WHATSAPP_WEBHOOK_VERIFICATION_TOKEN = os.getenv("WHATSAPP_WEBHOOK_VERIFICATION_TOKEN", "")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

RESTAURANT_BOT_URL = os.getenv(
    "RESTAURANT_BOT_URL",
    "https://restaurant-bot-production-a133.up.railway.app/webhook",
)

# Basic sanity checks on startup (fail loud, not silent)
missing = [k for k, v in {
    "WHATSAPP_TOKEN": WHATSAPP_TOKEN,
    "WHATSAPP_PHONE_NUMBER_ID": WHATSAPP_PHONE_NUMBER_ID,
    "WHATSAPP_APP_SECRET": WHATSAPP_APP_SECRET,
    "WHATSAPP_WEBHOOK_VERIFICATION_TOKEN": WHATSAPP_WEBHOOK_VERIFICATION_TOKEN,
    "DEEPGRAM_API_KEY": DEEPGRAM_API_KEY,
    "GROQ_API_KEY": GROQ_API_KEY,
}.items() if not v]
if missing:
    logger.warning(f"Missing env vars (bot will still start, but calls will fail): {missing}")

logger.remove()
logger.add(sys.stdout, level="INFO")


# ── Wild Bites menu knowledge for the voice AI ──────────────
# Mirrors what the restaurant-bot has, but simplified for voice.
MENU_FOR_VOICE = """
You are Alex, the friendly voice assistant for Wild Bites Restaurant.
You take orders over the phone via WhatsApp calling.

MENU HIGHLIGHTS (prices in USD):
- DEALS: Family Bundle $29.99, Duo Deal $18.99
- BURGERS: Classic Smash $8.99, Bacon Cheeseburger $10.99, Double Smash $12.99, Chicken Sandwich $9.99
- PIZZA: Margherita $13.99, Pepperoni $15.99, BBQ Chicken $16.99, Veggie Supreme $14.99
- BBQ: Half Rack Ribs $18.99, Full Rack Ribs $28.99, Brisket Plate $19.99
- FISH: Grilled Salmon $19.99, Fish & Chips $14.99, Shrimp Platter $17.99
- SIDES: Fries $3.99, Mac & Cheese $4.99, Coleslaw $2.99, 6 Wings $7.99, Nachos $6.99
- DRINKS: Coke $2.49, Pepsi $2.49, Shakes $4.99
- DESSERTS: Brownie $4.99, Chocolate Cake $5.99

ORDER RULES:
- Minimum $30 for delivery ($4.99 delivery fee)
- Minimum $10 for pickup
- 8% tax added automatically
"""

SYSTEM_PROMPT = f"""{MENU_FOR_VOICE}

YOUR PERSONALITY:
- Warm, friendly, upbeat — like a good waiter
- SHORT responses (1-2 sentences max, this is a voice call not a chat)
- Never read the full menu unless asked
- Ask ONE question at a time
- Suggest popular items if customer is unsure

CALL FLOW:
1. Greet warmly: "Hi! Thanks for calling Wild Bites. I'm Alex. What can I get started for you?"
2. Take their order one item at a time. Confirm each item briefly.
3. Suggest ONE upsell max (e.g., "Want to add fries or a drink?")
4. Ask: delivery or pickup?
5. If delivery: get address. If pickup: confirm pickup time (~25 min).
6. Get their name.
7. Ask for payment preference (cash or card-on-delivery).
8. Repeat the full order + total + ETA before confirming.
9. End with: "Your order is confirmed! You'll get a WhatsApp message shortly. Thanks for choosing Wild Bites!"

IMPORTANT:
- Keep every response SHORT (under 25 words). This is spoken aloud.
- Do not use markdown, emojis, or special formatting — you are speaking.
- If the caller asks something off-topic, politely redirect to ordering.
- If the caller is unclear, ask ONE clarifying question.
- Do not invent menu items — only use what's listed above.
"""


# ── Pipecat bot runner per call ──────────────────────────────
async def run_bot(webrtc_connection: SmallWebRTCConnection):
    """
    Called once per incoming WhatsApp voice call.
    Builds the pipeline: STT -> LLM -> TTS and runs until the call ends.
    """
    logger.info(f"Starting voice bot for call: {webrtc_connection.pc_id}")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        model="nova-2-general",
    )

    llm = GroqLLMService(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
    )

    tts = DeepgramTTSService(
        api_key=DEEPGRAM_API_KEY,
        voice="aura-asteria-en",
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Caller connected — greeting them")
        # Seed the first bot turn so caller hears greeting immediately
        messages.append({
            "role": "system",
            "content": "Greet the caller warmly and ask what they'd like to order.",
        })
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Caller disconnected — ending pipeline")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    logger.info(f"Voice bot ended for call: {webrtc_connection.pc_id}")


# ── FastAPI app + WhatsApp client ────────────────────────────
http_session: Optional[aiohttp.ClientSession] = None
whatsapp_client: Optional[WhatsAppClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_session, whatsapp_client
    http_session = aiohttp.ClientSession()
    whatsapp_client = WhatsAppClient(
        whatsapp_token=WHATSAPP_TOKEN,
        phone_number_id=WHATSAPP_PHONE_NUMBER_ID,
        session=http_session,
        whatsapp_secret=WHATSAPP_APP_SECRET,
    )
    logger.info("Voice bot ready. Waiting for WhatsApp calls...")
    yield
    if whatsapp_client:
        await whatsapp_client.terminate_all_calls()
    if http_session:
        await http_session.close()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"service": "wild-bites-voice-bot", "status": "ok"}


@app.get("/whatsapp")
async def verify_whatsapp_webhook(request: Request):
    """
    Meta webhook verification (GET) — happens once when you save the webhook URL.
    """
    params = dict(request.query_params)
    if whatsapp_client is None:
        return PlainTextResponse("Not ready", status_code=503)
    try:
        challenge = await whatsapp_client.handle_verify_webhook_request(
            params=params,
            expected_verification_token=WHATSAPP_WEBHOOK_VERIFICATION_TOKEN,
        )
        return PlainTextResponse(str(challenge))
    except Exception as e:
        logger.error(f"Webhook verification failed: {e}")
        return PlainTextResponse("Forbidden", status_code=403)


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    Meta calls webhook (POST).
    - 'connect' event: caller is dialing us -> spin up Pipecat bot + answer
    - 'terminate' event: caller hung up -> clean up
    """
    raw_body = await request.body()
    sha256_signature = request.headers.get("x-hub-signature-256", "")

    try:
        body_json = await request.json()
    except Exception as e:
        logger.error(f"Invalid JSON from Meta: {e}")
        return JSONResponse({"status": "bad_request"}, status_code=400)

    # Log non-call events (so messages-only webhooks don't crash)
    try:
        entry = body_json.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        if "calls" not in value:
            logger.info(f"Non-call webhook received (field={changes.get('field')}) — ignoring")
            return JSONResponse({"status": "ok"})
    except Exception:
        pass

    # Parse into Pipecat's typed webhook request
    try:
        webhook_request = WhatsAppWebhookRequest.model_validate(body_json)
    except Exception as e:
        logger.error(f"Could not parse Meta calling webhook: {e}")
        return JSONResponse({"status": "ignored"})

    if whatsapp_client is None:
        return JSONResponse({"status": "not_ready"}, status_code=503)

    # Pipecat handles the SDP handshake + pre-accept + accept + WebRTC setup.
    # `run_bot` is our callback — called after the WebRTC peer is ready.
    try:
        handled = await whatsapp_client.handle_webhook_request(
            request=webhook_request,
            connection_callback=run_bot,
            raw_body=raw_body,
            sha256_signature=sha256_signature,
        )
        logger.info(f"Webhook handled={handled}")
        return JSONResponse({"status": "ok"})
    except Exception as e:
        logger.error(f"Error handling call webhook: {e}")
        return JSONResponse({"status": "error"}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("bot:app", host="0.0.0.0", port=PORT, log_level="info")
