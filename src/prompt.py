"""Step 1 - what we ask the model, and the shape we want back."""

import logging
logger = logging.getLogger(__name__)

# The model gets these words and nothing else. Kept here so result.py and
# report.py can import them instead of keeping their own copies.
# No "Mismatch" here - you can't tell a mismatch from a photo, you need the
# spreadsheet. report.py works that one out

OBJECT_CATEGORIES = ("Carton", "Pallet", "Returnable_Container", "Crate", "Kit_Tray")
DAMAGE_TYPES = ("None", "Dent", "Crush", "Tear", "Water", "Corrosion", "Broken-Seal")
DAMAGE_SEVERITIES = ("None", "Minor", "Moderate", "Severe")
LABEL_STATES = ("Readable", "Partially-Readable", "Missing")
KIT_COMPLETENESS = ("N/A", "Complete", "Incomplete")
PALLET_STABILITY = ("N/A", "Stable", "Overhang", "Leaning", "Broken")


SYSTEM_PROMPT = f"""\
You are a goods-inwards inspector at an automotive parts warehouse. You are \
shown one or more photographs of a SINGLE inbound unit, and you report only \
what is visible.

You do NOT decide whether the unit is accepted, released, held or quarantined. \
Another part of the system does that by comparing your report against the \
shipment paperwork. Your report is evidence, not a verdict.

If several images are given they show the SAME unit from different angles. \
Report the most severe condition visible across all of them - if one face is \
torn, the unit is torn.

FIELDS - use only these exact values:

object_category   {" | ".join(OBJECT_CATEGORIES)}
  A cardboard box is Carton. A load on a wooden pallet is Pallet. A reusable \
plastic bin or tote (often blue or grey, e.g. a KLT) is Returnable_Container. \
A wooden box built from planks is Crate. A tray with compartments holding a \
set of items is Kit_Tray.

damage_type       {" | ".join(DAMAGE_TYPES)}
  The single most serious damage visible. Broken-Seal covers cut straps, \
snapped security ties, ripped VOID tape, and packaging slit open and re-taped. \
Water covers staining, sogginess, warping and mould. Corrosion is rust on the \
part itself.

damage_severity   {" | ".join(DAMAGE_SEVERITIES)}
  Minor     cosmetic only - a scuff, a small dent, a crease.
  Moderate  clearly past cosmetic - a deep crush, a tear that opens the unit.
  Severe    the unit has lost integrity, or the part itself is damaged.
  Use None only when damage_type is None.

label_state       {" | ".join(LABEL_STATES)}
  Readable            a label is present and you can read its identifier.
  Partially-Readable  a label is present but torn, smudged or obscured.
  Missing             no label is visible on any photographed face.

label_read_value
  The identifier exactly as printed - character for character, including \
brackets, hyphens and leading zeros. Do not tidy it up, correct it or fill in \
what you think is missing. Null if nothing is readable.

kit_completeness  {" | ".join(KIT_COMPLETENESS)}
  N/A unless the unit is a Kit_Tray. Incomplete when compartments are empty.

pallet_stability  {" | ".join(PALLET_STABILITY)}
  N/A unless the unit is a pallet or a palletised load.
  Overhang  the load extends past the pallet edge.
  Leaning   the stack is tilted or has shifted.
  Broken    a pallet board is cracked or splintered.

visible_evidence
  2 to 5 short phrases naming what you actually saw, each tied to something \
specific in the image: "dark water stain across the lower third of the front \
face", not "the box is damaged". A human reads these to check your work.

Judge only from the pixels. Never infer a part number, a supplier or a \
quantity that is not legible in the image. If you cannot see something, say so \
through the state fields - an admitted uncertainty is far cheaper here than a \
confident wrong reading, because a wrong reading releases goods."""


# Everything is required. Leave a field out and it falls back to a default,
# and the default for damage is "None" - which reads as "checked, it's fine".
RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["object_category", "damage_type", "damage_severity",
                 "label_state", "label_read_value", "kit_completeness",
                 "pallet_stability", "visible_evidence"],
    "properties": {
        "object_category":  {"type": "string", "enum": list(OBJECT_CATEGORIES)},
        "damage_type":      {"type": "string", "enum": list(DAMAGE_TYPES)},
        "damage_severity":  {"type": "string", "enum": list(DAMAGE_SEVERITIES)},
        "label_state":      {"type": "string", "enum": list(LABEL_STATES)},
        "label_read_value": {"type": ["string", "null"]},
        "kit_completeness": {"type": "string", "enum": list(KIT_COMPLETENESS)},
        "pallet_stability": {"type": "string", "enum": list(PALLET_STABILITY)},
        "visible_evidence": {"type": "array", "items": {"type": "string"},
                             "maxItems": 5},
    },
}


def build():
    """The prompt and the schema, for llmaas.py to send."""
    logger.debug("prompt %d chars, %d fields in the schema",
                 len(SYSTEM_PROMPT), len(RESPONSE_SCHEMA["properties"]))
    return SYSTEM_PROMPT, RESPONSE_SCHEMA


def userText(imageCount, hint=""):
    """The message that rides along with the images."""
    text = f"Inspect this inbound unit. {imageCount} image(s) of the SAME unit."
    if hint and hint.strip():
        logger.debug("operator hint attached: %r", hint.strip()[:80])
        # Operator context like "the pallet on bay 3". Don't ever put the
        # shipment record in here - if you tell the model what the label should
        # say, it'll agree with you, and those are the units we most need it to
        # disagree on.
        text += f"\n\nOperator note: {hint.strip()}"
    return text


if __name__ == "__main__":
    # python src/prompt.py - shows exactly what step 2 will send.
    import json
    import os
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s  %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S")

    systemPrompt, schema = build()
    print(systemPrompt)
    print("\n" + "-" * 70 + "\n")
    print(json.dumps(schema, indent=2))
    print("\n" + "-" * 70 + "\n")
    print(userText(2, "the pallet on bay 3"))
