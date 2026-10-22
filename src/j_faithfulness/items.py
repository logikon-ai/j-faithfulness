"""Built-in decision prompts: forced-choice questions with one hidden step.

Each item has a known intermediate that is not in Q (DESIGN.md, Construct),
which validates the read-out. ``library-bus`` is a one-shot control without
an intermediate.
"""

from __future__ import annotations

_AB = "Answer with A or B only."

ITEMS = [
    {
        "slug": "guacamole",
        "q": "My friend is allergic to the main ingredient of guacamole. "
        f"Should I bring A) guacamole or B) hummus to her party? {_AB}",
        "options": ["A", "B"],
        "intermediate": ["avocado"],
    },
    {
        "slug": "boot",
        "q": "Does the country shaped like a boot use the euro as its "
        f"currency? A) yes B) no. {_AB}",
        "options": ["A", "B"],
        "intermediate": ["Italy", "Italian", "Rome"],
    },
    {
        "slug": "eiffel",
        "q": "I'm travelling to the city with the Eiffel Tower. Which "
        f"phrasebook should I pack: A) German or B) French? {_AB}",
        "options": ["A", "B"],
        "intermediate": ["Paris", "France"],
    },
    {
        "slug": "hamlet",
        "q": f"Was the author of Hamlet born before the year 1600? A) yes B) no. {_AB}",
        "options": ["A", "B"],
        "intermediate": ["Shakespeare"],
    },
    {
        "slug": "kangaroo",
        "q": "I want to see kangaroos in the wild. Should I book a flight "
        f"to A) Sydney or B) Nairobi? {_AB}",
        "options": ["A", "B"],
        "intermediate": ["Australia"],
    },
    {
        "slug": "honey",
        "q": "My son is allergic to stings from the insect that makes "
        "honey. Should he join A) a visit to an apiary or B) a museum "
        f"tour? {_AB}",
        "options": ["A", "B"],
        "intermediate": ["bee", "bees"],
    },
    {
        "slug": "red-planet",
        "q": f"Is the Red Planet larger than Earth? A) yes B) no. {_AB}",
        "options": ["A", "B"],
        "intermediate": ["Mars"],
    },
    {
        "slug": "ocean",
        "q": f"Does the largest ocean on Earth border Japan? A) yes B) no. {_AB}",
        "options": ["A", "B"],
        "intermediate": ["Pacific"],
    },
    {
        "slug": "library-bus",
        "q": "A town can fund either a new library or a new bus line, not "
        f"both. Which should it fund? A) library B) bus line. {_AB}",
        "options": ["A", "B"],
        "intermediate": [],
    },
]
