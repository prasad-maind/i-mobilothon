"""Step 3 - take whatever the model said and turn it into something trustworthy.

The schema in prompt.py fixes the SHAPE of the reply, not the CONTENT. You can
get {"damage_type": "water damage"} inside perfectly valid JSON. If that got
through to report.py it would match nothing, fall out of every comparison, and
end up reading as "no damage found" - which is how a corroded part gets released.

So everything the model says gets checked against the word lists here before it
goes any further.

    python src/result.py              # run a deliberately messy reply through it
    python src/result.py reply.json   # or parse a real one
"""

import json
import logging
from dataclasses import dataclass, field, asdict

import prompt

logger = logging.getLogger(__name__)

# The spellings a model reaches for even when you hand it a list. Keys are
# already normalised - lowercase, no spaces or dashes.
DAMAGE_WORDS = {
    "none": "None", "nodamage": "None", "undamaged": "None", "clean": "None",
    "dent": "Dent", "dented": "Dent", "dents": "Dent", "ding": "Dent",
    "scuff": "Dent", "scratch": "Dent", "crease": "Dent",
    "crush": "Crush", "crushed": "Crush", "compressed": "Crush",
    "impact": "Crush", "collapsed": "Crush",
    "tear": "Tear", "torn": "Tear", "rip": "Tear", "ripped": "Tear",
    "puncture": "Tear", "hole": "Tear", "split": "Tear",
    "water": "Water", "waterdamage": "Water", "wet": "Water", "damp": "Water",
    "soggy": "Water", "moisture": "Water", "mold": "Water", "mould": "Water",
    "corrosion": "Corrosion", "corroded": "Corrosion", "rust": "Corrosion",
    "rusty": "Corrosion", "oxidation": "Corrosion",
    "brokenseal": "Broken-Seal", "seal": "Broken-Seal", "tamper": "Broken-Seal",
    "tampered": "Broken-Seal", "void": "Broken-Seal", "cutstrap": "Broken-Seal",
}

SEVERITY_WORDS = {
    "none": "None", "nil": "None",
    "minor": "Minor", "light": "Minor", "slight": "Minor", "cosmetic": "Minor",
    "small": "Minor", "superficial": "Minor",
    "moderate": "Moderate", "medium": "Moderate", "significant": "Moderate",
    "severe": "Severe", "major": "Severe", "heavy": "Severe",
    "critical": "Severe", "extensive": "Severe",
}

LABEL_WORDS = {
    "readable": "Readable", "legible": "Readable", "clear": "Readable",
    "visible": "Readable", "present": "Readable", "ok": "Readable",
    "partiallyreadable": "Partially-Readable", "partial": "Partially-Readable",
    "partiallylegible": "Partially-Readable", "obscured": "Partially-Readable",
    "faded": "Partially-Readable", "smudged": "Partially-Readable",
    "missing": "Missing", "absent": "Missing", "none": "Missing",
    "nolabel": "Missing", "notvisible": "Missing", "notpresent": "Missing",
}

KIT_WORDS = {
    "na": "N/A", "notapplicable": "N/A", "unknown": "N/A",
    "complete": "Complete", "full": "Complete", "allpresent": "Complete",
    "incomplete": "Incomplete", "partial": "Incomplete", "short": "Incomplete",
    "missingitems": "Incomplete", "empty": "Incomplete",
}

STABILITY_WORDS = {
    "na": "N/A", "notapplicable": "N/A", "unknown": "N/A",
    "stable": "Stable", "ok": "Stable", "secure": "Stable", "upright": "Stable",
    "overhang": "Overhang", "overhanging": "Overhang", "protruding": "Overhang",
    "leaning": "Leaning", "tilted": "Leaning", "tilting": "Leaning",
    "unstable": "Leaning", "shifted": "Leaning",
    "broken": "Broken", "splintered": "Broken", "cracked": "Broken",
    "collapsed": "Broken",
}

CATEGORY_WORDS = {
    "carton": "Carton", "box": "Carton", "cardboardbox": "Carton", "case": "Carton",
    "pallet": "Pallet", "europallet": "Pallet", "palletload": "Pallet",
    "skid": "Pallet", "palletisedload": "Pallet",
    "returnablecontainer": "Returnable_Container", "kltbin": "Returnable_Container",
    "klt": "Returnable_Container", "bin": "Returnable_Container",
    "tote": "Returnable_Container", "plasticbin": "Returnable_Container",
    "crate": "Crate", "woodencrate": "Crate", "woodbox": "Crate",
    "kittray": "Kit_Tray", "tray": "Kit_Tray", "kit": "Kit_Tray",
    "blistertray": "Kit_Tray",
}


@dataclass
class Observation:
    """What the camera saw. One of these per unit, not per photo.

    The defaults say "we didn't look", not "we looked and it's fine" - those two
    have to stay tellable apart downstream.
    """
    objectCategory: str = None
    damageType: str = "None"
    damageSeverity: str = "None"
    labelState: str = "Missing"
    labelReadValue: str = None
    kitCompleteness: str = "N/A"
    palletStability: str = "N/A"
    visibleEvidence: list = field(default_factory=list)
    imageCount: int = 0

    def asDict(self):
        return asdict(self)


def _normalise(value):
    """'Broken Seal' -> 'brokenseal', so every spelling lands on one key."""
    return "".join(str(value).lower().split()).replace("-", "").replace("_", "")


def _coerceOne(value, allowed, synonyms, default, fieldName):
    """Force one field into the word list. Exact, then case, then synonym, then give up."""
    if value is None:
        return default

    text = str(value).strip()
    if text in allowed:
        return text

    for candidate in allowed:
        if candidate.lower() == text.lower():
            return candidate

    mapped = synonyms.get(_normalise(text))
    if mapped and mapped in allowed:
        logger.info("%s: %r -> %r", fieldName, text[:40], mapped)
        return mapped

    # Nothing matched. Falling back is safe, but it's worth shouting about - a
    # field that lands here often means the prompt needs a word adding.
    logger.warning("%s: %r isn't in the vocabulary, using %r",
                   fieldName, text[:40], default)
    return default


def parse(raw):
    """The model's dict -> a clean Observation."""
    raw = raw or {}

    # Category is the one field that stays None when unknown. Everything else
    # has a sensible "nothing to report" value; forcing a guess here would have
    # us assert a box is a Carton when nobody actually saw one.
    category = raw.get("object_category")
    if category and str(category).strip():
        category = _coerceOne(category, prompt.OBJECT_CATEGORIES, CATEGORY_WORDS,
                              "", "object_category") or None
    else:
        category = None

    damage = _coerceOne(raw.get("damage_type"), prompt.DAMAGE_TYPES,
                        DAMAGE_WORDS, "None", "damage_type")
    severity = _coerceOne(raw.get("damage_severity"), prompt.DAMAGE_SEVERITIES,
                          SEVERITY_WORDS, "None", "damage_severity")

    # The model has contradicted itself. Read it the careful way round: a crush
    # reported as "no severity" is still a crush.
    if damage != "None" and severity == "None":
        logger.info("%s damage with no severity, calling it Moderate", damage)
        severity = "Moderate"
    if damage == "None" and severity != "None":
        logger.info("severity %s but no damage type, calling it a Dent", severity)
        damage = "Dent"

    labelState = _coerceOne(raw.get("label_state"), prompt.LABEL_STATES,
                            LABEL_WORDS, "Missing", "label_state")

    readValue = raw.get("label_read_value")
    readValue = str(readValue).strip() if readValue else None
    if readValue and readValue.lower() in ("none", "null", "n/a", "unknown", ""):
        readValue = None
    # A label you can't actually read isn't readable, whatever the model called it.
    if labelState in ("Readable", "Partially-Readable") and not readValue:
        logger.info("label_state=%s but nothing was read, calling it Missing",
                    labelState)
        labelState = "Missing"

    evidence = raw.get("visible_evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    evidence = [str(item).strip() for item in evidence if str(item).strip()][:5]

    observation = Observation(
        objectCategory=category,
        damageType=damage,
        damageSeverity=severity,
        labelState=labelState,
        labelReadValue=readValue,
        kitCompleteness=_coerceOne(raw.get("kit_completeness"),
                                   prompt.KIT_COMPLETENESS, KIT_WORDS,
                                   "N/A", "kit_completeness"),
        palletStability=_coerceOne(raw.get("pallet_stability"),
                                   prompt.PALLET_STABILITY, STABILITY_WORDS,
                                   "N/A", "pallet_stability"),
        visibleEvidence=evidence,
        imageCount=int(raw.get("image_count") or 0),
    )

    logger.info("observation: %s / %s %s / label %s %r",
                observation.objectCategory, observation.damageSeverity,
                observation.damageType, observation.labelState,
                (observation.labelReadValue or "")[:24])
    return observation


# A reply with something wrong in every field, for the standalone run below.
MESSY_EXAMPLE = {
    "object_category": "cardboard box",
    "damage_type": "water damage",
    "damage_severity": "",
    "label_state": "partially legible",
    "label_read_value": "  (01)0000000100037  ",
    "kit_completeness": "not applicable",
    "pallet_stability": "unknown",
    "visible_evidence": ["dark stain across the lower front face", "  ", "corner scuffed"],
    "image_count": 2,
}


def main(argv):
    """Standalone: a JSON reply in, a clean Observation out."""
    if len(argv) > 1:
        raw = json.loads(open(argv[1], encoding="utf-8").read())
        print(f"reading   {argv[1]}")
    else:
        raw = MESSY_EXAMPLE
        print("no file given, using the built-in messy example")

    print("\nwhat came in:")
    print(json.dumps(raw, indent=2))
    print("\nrunning it through parse()...\n")

    observation = parse(raw)

    print("\nwhat comes out:")
    print(json.dumps(observation.asDict(), indent=2))
    return 0


if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s  %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S")
    sys.exit(main(sys.argv))
