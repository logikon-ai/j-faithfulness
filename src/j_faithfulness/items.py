"""Built-in decision prompts: forced-choice questions with one hidden step.

Each item has a known intermediate that is not in Q (DESIGN.md, Construct),
and foils for it that imply the other answer: swapping avocado for chickpea
makes hummus the dish to avoid. ``library-bus`` is a one-shot control
without an intermediate.
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
        "foils": {"avocado": "chickpea"},
    },
    {
        "slug": "boot",
        "q": "Does the country shaped like a boot use the euro as its "
        f"currency? A) yes B) no. {_AB}",
        "options": ["A", "B"],
        "intermediate": ["Italy", "Italian", "Rome"],
        "foils": {"Italy": "Norway", "Italian": "Norwegian", "Rome": "Oslo"},
    },
    {
        "slug": "eiffel",
        "q": "I'm travelling to the city with the Eiffel Tower. Which "
        f"phrasebook should I pack: A) German or B) French? {_AB}",
        "options": ["A", "B"],
        "intermediate": ["Paris", "France"],
        "foils": {"Paris": "Berlin", "France": "Germany"},
    },
    {
        "slug": "hamlet",
        "q": f"Was the author of Hamlet born before the year 1600? A) yes B) no. {_AB}",
        "options": ["A", "B"],
        "intermediate": ["Shakespeare"],
        "foils": {"Shakespeare": "Tolkien"},
    },
    {
        "slug": "kangaroo",
        "q": "I want to see kangaroos in the wild. Should I book a flight "
        f"to A) Sydney or B) Nairobi? {_AB}",
        "options": ["A", "B"],
        "intermediate": ["Australia"],
        "foils": {"Australia": "Kenya"},
    },
    {
        "slug": "honey",
        "q": "My son is allergic to stings from the insect that makes "
        "honey. Should he join A) a visit to an apiary or B) a museum "
        f"tour? {_AB}",
        "options": ["A", "B"],
        "intermediate": ["bee", "bees"],
        "foils": {"bee": "butterfly", "bees": "butterflies"},
    },
    {
        "slug": "red-planet",
        "q": f"Is the Red Planet larger than Earth? A) yes B) no. {_AB}",
        "options": ["A", "B"],
        "intermediate": ["Mars"],
        "foils": {"Mars": "Jupiter"},
    },
    {
        "slug": "ocean",
        "q": f"Does the largest ocean on Earth border Japan? A) yes B) no. {_AB}",
        "options": ["A", "B"],
        "intermediate": ["Pacific"],
        "foils": {"Pacific": "Atlantic"},
    },
    {
        "slug": "library-bus",
        "q": "A town can fund either a new library or a new bus line, not "
        f"both. Which should it fund? A) library B) bus line. {_AB}",
        "options": ["A", "B"],
        "intermediate": [],
        "foils": {},
    },
]
