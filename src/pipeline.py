"""The four steps, in order. This is the whole application.

    1. prompt   what to ask
    2. llmaas   ask it
    3. result   check the answer is made of real words
    4. report   fetch the paperwork, compare, decide

Nothing in here knows about Flask. It takes bytes and gives back a dict, so the
web app and the command line run exactly the same code.

    python src/pipeline.py SHP-2026-00001 photo.jpg
    python src/pipeline.py auto photo.jpg        # work the shipment out from the label
"""

import logging
import time

import llmaas
import prompt
import report
import result

logger = logging.getLogger(__name__)


def run(images, shipmentId=None, hint=""):
    """Photos plus (optionally) a shipment ID, in. The finished answer, out.

    Leave shipmentId out and report.build() works it out from the label the
    camera read.
    """
    started = time.time()
    blind = False
    blindReason = ""

    # 1 + 2 - ask the model what it can see.
    try:
        systemPrompt, schema = prompt.build()
        userMessage = prompt.userText(len(images), hint)
        reply = llmaas.look(images, systemPrompt, schema, userMessage)
        reply["image_count"] = len(images)
    except llmaas.Unavailable as error:
        # Can't see. Don't guess, and don't blow up - carry on with the half we
        # do have and say so plainly. The zone check never needed the photo.
        logger.warning("no observation for %s: %s", shipmentId or "?", error)
        blind, blindReason, reply = True, str(error), {}

    # 3 - make sure every value is one we actually recognise.
    observation = result.Observation() if blind else result.parse(reply)

    # 4 - the spreadsheet, the comparison and the decision.
    answer = report.build(observation, shipmentId, blind=blind)
    answer["blind"] = blind
    answer["blindReason"] = blindReason
    answer["elapsed"] = round(time.time() - started, 2)

    logger.info("%s -> %s in %.1fs%s",
                answer.get("Shipment_ID", "?"),
                answer.get("Expected_Decision", answer.get("error")),
                answer["elapsed"],
                "  (blind - no photo assessed)" if blind else "")
    return answer


def main(argv):
    """Standalone: a shipment ID (or 'auto') and some image paths."""
    from pathlib import Path

    if len(argv) < 2:
        print("Usage: python src/pipeline.py SHP-2026-00001 [photo.jpg ...]")
        print("       python src/pipeline.py auto photo.jpg")
        print("\nWith no images it just reads the spreadsheet.")
        return 1

    shipmentId = None if argv[1].lower() == "auto" else argv[1]
    images = [Path(p).read_bytes() for p in argv[2:]]

    if images:
        print(f"{len(images)} image(s), shipment {shipmentId or '(work it out)'}\n")
        answer = run(images, shipmentId)
    else:
        print(f"no images - reading the record only, {shipmentId}\n")
        answer = report.build(result.Observation(), shipmentId)

    print(report.asText(answer))
    print()
    return 0


if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s  %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    sys.exit(main(sys.argv))
