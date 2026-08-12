"""JSON generation: synthetic prompts asking for a JSON object with a
specific set of fields and types, scored on schema conformance.

There's no off-the-shelf benchmark for this (per the README this is a custom
task), so prompts are generated programmatically from a small pool of schema
templates + subject words, deterministically seeded so a pilot n=50 run is
the first 50 items of a full n=500 run, same convention as the other tasks.
Scoring only checks structure — does the output parse as JSON, are all
required fields present, do their types match — not content, since there's
no single "correct" person/product/etc. to generate.
"""

import json
import random
import re

from config import SEED

SCHEMAS = [
    {"name": "person", "fields": {"name": "string", "age": "integer", "email": "string", "is_student": "boolean"}},
    {"name": "product", "fields": {"name": "string", "price": "number", "in_stock": "boolean", "tags": "array"}},
    {"name": "event", "fields": {"title": "string", "date": "string", "attendee_count": "integer", "is_virtual": "boolean"}},
    {"name": "recipe", "fields": {"name": "string", "prep_time_minutes": "integer", "ingredients": "array", "vegetarian": "boolean"}},
    {"name": "book", "fields": {"title": "string", "author": "string", "pages": "integer", "genres": "array"}},
]

SUBJECTS = [
    "a science fiction novel", "a local bakery's best-selling item", "a company offsite",
    "a weeknight dinner", "a mystery paperback", "an indie video game", "a neighborhood coffee shop",
    "a university lecture series", "a hiking trail guide entry", "a podcast episode",
    "a startup's flagship product", "a weekend farmers market stall", "a documentary film",
    "a board game", "a fitness class", "a museum exhibit", "a food truck",
    "a children's picture book", "a smartphone app", "a music festival act",
]

_TYPE_LABEL = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list}


def _prompt(schema: dict, subject: str) -> str:
    field_list = ", ".join(f"{name} ({typ})" for name, typ in schema["fields"].items())
    return (
        f"Generate a JSON object describing {subject}. "
        f"It must have exactly these fields: {field_list}. "
        f"Respond with ONLY the JSON object — no explanation, no markdown code fences."
    )


def score(output: str, gold: dict) -> bool:
    """gold is a {field_name: type_name} dict. All fields must be present
    with a value of the matching Python type; extra fields are tolerated.

    bool is a subclass of int in Python, so an "integer" field is checked
    against int-but-not-bool specifically — otherwise `true`/`false` would
    incorrectly pass as a valid integer.
    """
    text = re.sub(r"^```(?:json)?|```$", "", output.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(obj, dict):
        return False

    for field, type_name in gold.items():
        if field not in obj:
            return False
        value = obj[field]
        if type_name == "integer" and isinstance(value, bool):
            return False
        if not isinstance(value, _TYPE_LABEL[type_name]):
            return False
    return True


def load_items(n: int) -> list[dict]:
    rng = random.Random(SEED)
    items = []
    for i in range(n):
        schema = rng.choice(SCHEMAS)
        subject = rng.choice(SUBJECTS)
        items.append({
            "id": f"json-{i}",
            "messages": [{"role": "user", "content": _prompt(schema, subject)}],
            "gold": schema["fields"],
        })
    return items
