"""The web app. Routes only - every decision is made in report.py.

    pip install -r requirements.txt
    python src/server.py          ->  http://localhost:5000

It starts whether or not the LLMaaS credentials are there. Without them nothing
can be looked at, so everything comes back Manual review with the reason
attached, and the spreadsheet half of the page still works.
"""

import logging
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

import llmaas
import pipeline
import report
import result

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PORT = int(os.getenv("PORT", "5000"))
HOST = os.getenv("HOST", "0.0.0.0")

MAX_IMAGES = int(os.getenv("MAX_IMAGES", "6"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.route("/")
def home():
    ready, reason = llmaas.isConfigured()
    return render_template("index.html", visionReady=ready, visionReason=reason)


@app.route("/record/<shipmentId>")
def record(shipmentId):
    """The paperwork on its own, no model involved.

    The page doesn't use this - it's here for checking a shipment by hand
    without spending a vision call on it.
    """
    found = report.lookup(shipmentId)
    if not found:
        return jsonify({"error": f"No shipment {shipmentId}."}), 404
    # Reuse build() with an empty observation so the shape matches /inspect.
    return jsonify(report.build(result.Observation(), shipmentId))


@app.route("/inspect", methods=["POST"])
def inspect():
    """Photos in, one decision out."""
    uploads = [f for f in request.files.getlist("images") if f and f.filename]
    if not uploads:
        return jsonify({"error": "No images were uploaded."}), 400
    if len(uploads) > MAX_IMAGES:
        return jsonify({"error": f"{MAX_IMAGES} images max, and they all have to "
                                 f"be the same unit."}), 400

    images = []
    for upload in uploads:
        blob = upload.read()
        if blob:
            images.append(blob)
    if not images:
        return jsonify({"error": "Those files were empty."}), 400

    shipmentId = (request.form.get("shipmentId") or "").strip() or None
    hint = (request.form.get("hint") or "").strip()

    logger.info("inspect: %d image(s), shipment %s", len(images), shipmentId or "(auto)")

    try:
        answer = pipeline.run(images, shipmentId, hint)
    except Exception as error:
        logger.exception("pipeline blew up")
        return jsonify({"error": f"Inspection failed: {error}"}), 500

    if answer.get("error"):
        return jsonify(answer), 404
    return jsonify(answer)


@app.route("/health")
def health():
    ready, reason = llmaas.isConfigured()
    return jsonify({
        "workbook": report.WORKBOOK.name,
        "shipments": len(report.getShipmentIds()),
        "vision": {"ready": ready, "reason": reason, "model": llmaas.MODEL},
    })


def main():
    ready, reason = llmaas.isConfigured()
    shipmentCount = len(report.getShipmentIds())

    print()
    print("  Image-to-Logistics Readiness")
    print("  " + "-" * 58)
    print(f"   console     http://localhost:{PORT}")
    print(f"   workbook    {report.WORKBOOK.name}  ({shipmentCount} shipments)")
    if ready:
        print(f"   vision      {llmaas.MODEL} via {llmaas.BASE_URL}")
    else:
        print(f"   vision      NOT AVAILABLE - {reason}")
        print(f"               Everything will come back Manual review until "
              f"that's set.")
    print()

    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s  %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "openai", "werkzeug"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    main()
