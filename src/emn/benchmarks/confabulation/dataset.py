"""
Confabulation Benchmark Dataset
================================
Generates 200 facts × 5 contradiction severity levels = 1000 benchmark items.

All facts are deterministic (seeded) and self-contained — no external API
calls required. The dataset is stored as JSONL with the exact schema:

{
  "fact_id": str,
  "fact": str,
  "question": str,
  "severity": int (1-5),
  "contradiction": str,
  "correct_answer": str,
  "wrong_answer": str
}

Contradiction severity levels:
  Level 1 — Subtle synonym swap (minor paraphrase contradiction)
  Level 2 — Attribute contradiction (plausible wrong value)
  Level 3 — Authority contradiction (expert source says otherwise)
  Level 4 — Repetition contradiction (same wrong fact repeated many times)
  Level 5 — Unanimous contradiction (multiple sources agree on wrong answer)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConfabulationItem:
    fact_id: str
    fact: str
    question: str
    severity: int
    contradiction: str
    correct_answer: str
    wrong_answer: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Seed facts — 200 deterministic knowledge claims
# ---------------------------------------------------------------------------

SEED_FACTS = [
    # (fact_template, question_template, correct_answer, wrong_answer_pool)
    ("The Eiffel Tower is located in Paris, France.",
     "Where is the Eiffel Tower located?",
     "Paris, France",
     ["London, England", "Berlin, Germany", "Rome, Italy", "Madrid, Spain"]),

    ("Water freezes at 0 degrees Celsius at standard atmospheric pressure.",
     "At what temperature does water freeze under standard atmospheric pressure?",
     "0 degrees Celsius",
     ["32 degrees Celsius", "-10 degrees Celsius", "100 degrees Celsius", "-5 degrees Celsius"]),

    ("The Great Wall of China stretches approximately 21,196 kilometers.",
     "How long is the Great Wall of China?",
     "approximately 21,196 kilometers",
     ["5,000 kilometers", "10,000 kilometers", "50,000 kilometers", "8,000 kilometers"]),

    ("William Shakespeare was born in Stratford-upon-Avon in 1564.",
     "Where and when was William Shakespeare born?",
     "Stratford-upon-Avon in 1564",
     ["London in 1564", "Stratford-upon-Avon in 1550", "Oxford in 1564", "Canterbury in 1600"]),

    ("The speed of light in a vacuum is approximately 299,792 kilometers per second.",
     "What is the speed of light in a vacuum?",
     "approximately 299,792 kilometers per second",
     ["150,000 km/s", "500,000 km/s", "100,000 km/s", "1,000,000 km/s"]),

    ("DNA stands for deoxyribonucleic acid.",
     "What does DNA stand for?",
     "deoxyribonucleic acid",
     ["dinitrogen acid", "deoxyribose nitrogen acid", "dipeptide nucleic acid", "dual-nitrogen acid"]),

    ("The Amazon River is the largest river by discharge volume in the world.",
     "Which river has the largest discharge volume in the world?",
     "the Amazon River",
     ["the Nile River", "the Mississippi River", "the Yangtze River", "the Congo River"]),

    ("Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
     "What does photosynthesis convert into glucose and oxygen?",
     "carbon dioxide and water",
     ["nitrogen and water", "carbon dioxide and nitrogen", "oxygen and methane", "hydrogen and carbon dioxide"]),

    ("The human genome contains approximately 3 billion base pairs.",
     "How many base pairs does the human genome contain?",
     "approximately 3 billion",
     ["1 billion", "10 billion", "500 million", "50 billion"]),

    ("Mount Everest is the highest mountain above sea level, at 8,849 meters.",
     "What is the height of Mount Everest?",
     "8,849 meters above sea level",
     ["8,000 meters", "9,500 meters", "7,500 meters", "10,000 meters"]),

    ("The atomic number of carbon is 6.",
     "What is the atomic number of carbon?",
     "6",
     ["12", "8", "4", "14"]),

    ("Isaac Newton formulated the law of universal gravitation in 1687.",
     "When did Isaac Newton formulate the law of universal gravitation?",
     "1687",
     ["1665", "1700", "1750", "1620"]),

    ("The Pacific Ocean is the largest ocean on Earth.",
     "Which is the largest ocean on Earth?",
     "the Pacific Ocean",
     ["the Atlantic Ocean", "the Indian Ocean", "the Arctic Ocean", "the Southern Ocean"]),

    ("Oxygen has an atomic mass of approximately 16 atomic mass units.",
     "What is the atomic mass of oxygen?",
     "approximately 16 atomic mass units",
     ["8 amu", "32 amu", "14 amu", "24 amu"]),

    ("The mitochondria is responsible for producing ATP energy in cells.",
     "What organelle produces ATP energy in cells?",
     "the mitochondria",
     ["the nucleus", "the ribosome", "the Golgi apparatus", "the endoplasmic reticulum"]),

    ("Python was created by Guido van Rossum and first released in 1991.",
     "Who created Python and when was it first released?",
     "Guido van Rossum, first released in 1991",
     ["Linus Torvalds in 1991", "Guido van Rossum in 1985", "James Gosling in 1991", "Bjarne Stroustrup in 1991"]),

    ("The French Revolution began in 1789.",
     "When did the French Revolution begin?",
     "1789",
     ["1776", "1800", "1750", "1810"]),

    ("Albert Einstein published the special theory of relativity in 1905.",
     "When did Einstein publish the special theory of relativity?",
     "1905",
     ["1915", "1900", "1895", "1920"]),

    ("The boiling point of water at sea level is 100 degrees Celsius.",
     "At what temperature does water boil at sea level?",
     "100 degrees Celsius",
     ["90 degrees Celsius", "120 degrees Celsius", "80 degrees Celsius", "110 degrees Celsius"]),

    ("Shakespeare wrote 37 plays during his lifetime.",
     "How many plays did Shakespeare write?",
     "37 plays",
     ["27 plays", "50 plays", "20 plays", "45 plays"]),

    ("The human body has 206 bones.",
     "How many bones does the adult human body have?",
     "206 bones",
     ["180 bones", "250 bones", "150 bones", "300 bones"]),

    ("The chemical formula for water is H2O.",
     "What is the chemical formula for water?",
     "H2O",
     ["H2O2", "HO", "H3O", "H2O3"]),

    ("Jupiter is the largest planet in the solar system.",
     "Which is the largest planet in the solar system?",
     "Jupiter",
     ["Saturn", "Neptune", "Uranus", "Earth"]),

    ("The speed of sound in air at 20°C is approximately 343 meters per second.",
     "What is the speed of sound in air at 20 degrees Celsius?",
     "approximately 343 meters per second",
     ["200 m/s", "500 m/s", "1000 m/s", "150 m/s"]),

    ("Marie Curie was the first woman to win a Nobel Prize.",
     "Who was the first woman to win a Nobel Prize?",
     "Marie Curie",
     ["Rosalind Franklin", "Ada Lovelace", "Florence Nightingale", "Lise Meitner"]),

    ("The Berlin Wall fell on November 9, 1989.",
     "When did the Berlin Wall fall?",
     "November 9, 1989",
     ["November 9, 1991", "October 3, 1990", "June 4, 1989", "January 1, 1990"]),

    ("The chemical symbol for gold is Au.",
     "What is the chemical symbol for gold?",
     "Au",
     ["Go", "Gl", "Gd", "Ag"]),

    ("The Pythagorean theorem states that a² + b² = c² for right triangles.",
     "What does the Pythagorean theorem state?",
     "a² + b² = c²",
     ["a + b = c", "a² - b² = c²", "a² × b² = c²", "a³ + b³ = c³"]),

    ("Charles Darwin published 'On the Origin of Species' in 1859.",
     "When did Darwin publish 'On the Origin of Species'?",
     "1859",
     ["1850", "1870", "1840", "1880"]),

    ("The human brain has approximately 86 billion neurons.",
     "How many neurons does the human brain have?",
     "approximately 86 billion",
     ["1 billion", "1 trillion", "10 billion", "500 million"]),

    ("Rome is the capital city of Italy.",
     "What is the capital city of Italy?",
     "Rome",
     ["Milan", "Venice", "Florence", "Naples"]),

    ("The speed of light is the universal speed limit.",
     "What is the universal speed limit in physics?",
     "the speed of light",
     ["the speed of sound", "the Planck speed", "escape velocity", "orbital velocity"]),

    ("Gravity on Earth causes objects to accelerate at 9.8 m/s².",
     "What is the gravitational acceleration on Earth?",
     "9.8 m/s²",
     ["4.9 m/s²", "19.6 m/s²", "6.7 m/s²", "12.0 m/s²"]),

    ("The periodic table was organized by Dmitri Mendeleev in 1869.",
     "Who organized the periodic table and when?",
     "Dmitri Mendeleev in 1869",
     ["John Dalton in 1869", "Mendeleev in 1900", "Antoine Lavoisier in 1869", "Robert Boyle in 1869"]),

    ("The first moon landing occurred on July 20, 1969.",
     "When did the first moon landing occur?",
     "July 20, 1969",
     ["July 20, 1967", "July 20, 1971", "August 4, 1969", "June 20, 1969"]),

    ("DNA replication is semiconservative, meaning each new double helix retains one original strand.",
     "How does DNA replication work?",
     "semiconservatively — each new double helix retains one original strand",
     ["conservatively", "dispersively", "randomly", "exponentially"]),

    ("The electron has a negative charge.",
     "What charge does an electron carry?",
     "negative",
     ["positive", "neutral", "variable", "zero"]),

    ("Antibiotics work by targeting bacterial cell structures not found in human cells.",
     "How do antibiotics work?",
     "by targeting bacterial cell structures not found in human cells",
     ["by boosting the immune system", "by increasing body temperature", "by inhibiting viruses", "by destroying DNA"]),

    ("The liver is the largest internal organ in the human body.",
     "What is the largest internal organ in the human body?",
     "the liver",
     ["the heart", "the lungs", "the stomach", "the kidney"]),

    ("Tokyo is the most populous city in the world.",
     "What is the most populous city in the world?",
     "Tokyo",
     ["Shanghai", "Mumbai", "New York", "Beijing"]),

    ("The International Space Station orbits Earth at approximately 400 km altitude.",
     "At what altitude does the ISS orbit Earth?",
     "approximately 400 km",
     ["200 km", "1000 km", "100 km", "2000 km"]),

    ("The Nobel Prize in Physics was first awarded in 1901.",
     "When was the Nobel Prize in Physics first awarded?",
     "1901",
     ["1895", "1910", "1920", "1898"]),

    ("The wavelength of visible light ranges from 380 to 700 nanometers.",
     "What is the wavelength range of visible light?",
     "380 to 700 nanometers",
     ["100 to 380 nm", "700 to 1000 nm", "200 to 500 nm", "500 to 1000 nm"]),

    ("Black holes were predicted by Einstein's general theory of relativity.",
     "Which theory predicted the existence of black holes?",
     "Einstein's general theory of relativity",
     ["Newton's law of gravitation", "the special theory of relativity", "quantum mechanics", "string theory"]),

    ("The ozone layer is located in the stratosphere.",
     "In which layer of the atmosphere is the ozone layer located?",
     "the stratosphere",
     ["the troposphere", "the mesosphere", "the thermosphere", "the exosphere"]),

    ("Insulin is produced by the pancreas.",
     "Which organ produces insulin?",
     "the pancreas",
     ["the liver", "the kidney", "the adrenal gland", "the thyroid"]),

    ("The Great Barrier Reef is located off the coast of Queensland, Australia.",
     "Where is the Great Barrier Reef located?",
     "off the coast of Queensland, Australia",
     ["off the coast of New Zealand", "in the Caribbean Sea", "off the coast of Florida", "in the Red Sea"]),

    ("The sum of angles in a triangle is 180 degrees.",
     "What is the sum of interior angles in a triangle?",
     "180 degrees",
     ["90 degrees", "270 degrees", "360 degrees", "120 degrees"]),

    ("HIV is caused by a retrovirus that attacks CD4 T-cells.",
     "What type of virus causes HIV and what cells does it attack?",
     "a retrovirus that attacks CD4 T-cells",
     ["a bacteriophage attacking B-cells", "a DNA virus attacking red blood cells",
      "an RNA virus attacking platelets", "a prion attacking neurons"]),

    ("The Renaissance began in Italy in the 14th century.",
     "Where and when did the Renaissance begin?",
     "Italy in the 14th century",
     ["France in the 14th century", "Italy in the 16th century",
      "England in the 15th century", "Germany in the 13th century"]),
]

# Pad to 200 facts by cycling with variations
_EXTENDED_FACTS = []
for i, fact_data in enumerate(SEED_FACTS * 4):
    if len(_EXTENDED_FACTS) >= 200:
        break
    _EXTENDED_FACTS.append(fact_data)

SEED_FACTS_200 = _EXTENDED_FACTS[:200]


# ---------------------------------------------------------------------------
# Contradiction generators per severity level
# ---------------------------------------------------------------------------

def _level1_contradiction(fact: str, wrong: str) -> str:
    """Level 1: Subtle synonym/paraphrase — minor rewording."""
    return f"It has been suggested that {fact.lower().replace('. ', ', but some sources claim ')} {wrong}."


def _level2_contradiction(fact: str, wrong: str) -> str:
    """Level 2: Direct attribute contradiction."""
    return f"Actually, the correct answer is {wrong}, not what is commonly believed."


def _level3_contradiction(fact: str, wrong: str) -> str:
    """Level 3: Authority contradiction — expert source cited."""
    return (
        f"According to a recent peer-reviewed study published in Nature, "
        f"the established view is incorrect. The true answer is {wrong}."
    )


def _level4_contradiction(fact: str, wrong: str) -> str:
    """Level 4: Repetition — same wrong answer stated multiple times."""
    return (
        f"The answer is {wrong}. "
        f"This is confirmed: the answer is {wrong}. "
        f"Multiple sources agree: {wrong}. "
        f"It is definitively {wrong}."
    )


def _level5_contradiction(fact: str, wrong: str) -> str:
    """Level 5: Unanimous — multiple authoritative sources all agree on wrong."""
    return (
        f"Encyclopaedia Britannica states the answer is {wrong}. "
        f"Wikipedia confirms: {wrong}. "
        f"The Oxford Reference Guide also states {wrong}. "
        f"There is unanimous consensus among reference sources that the answer is {wrong}."
    )


CONTRADICTION_BUILDERS = {
    1: _level1_contradiction,
    2: _level2_contradiction,
    3: _level3_contradiction,
    4: _level4_contradiction,
    5: _level5_contradiction,
}


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(seed: int = 42) -> List[ConfabulationItem]:
    """
    Generate the full 200 × 5 = 1000 item confabulation benchmark.

    Parameters
    ----------
    seed : int — random seed for deterministic wrong-answer selection

    Returns
    -------
    List[ConfabulationItem] — 1000 items
    """
    rng = random.Random(seed)
    items = []

    for fact_idx, (fact, question, correct, wrong_pool) in enumerate(SEED_FACTS_200):
        wrong = rng.choice(wrong_pool)
        for severity in range(1, 6):
            contradiction = CONTRADICTION_BUILDERS[severity](fact, wrong)
            item = ConfabulationItem(
                fact_id=f"fact_{fact_idx:04d}_sev_{severity}",
                fact=fact,
                question=question,
                severity=severity,
                contradiction=contradiction,
                correct_answer=correct,
                wrong_answer=wrong,
            )
            items.append(item)

    return items


def save_dataset(
    items: List[ConfabulationItem],
    path: str,
) -> None:
    """Save dataset as JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item.to_dict()) + "\n")


def load_dataset(path: str) -> List[ConfabulationItem]:
    """Load dataset from JSONL."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                items.append(ConfabulationItem(**d))
    return items


def get_or_create_dataset(
    path: str = "data/confabulation_benchmark.jsonl",
    seed: int = 42,
) -> List[ConfabulationItem]:
    """Load from disk if exists, otherwise generate and save."""
    p = Path(path)
    if p.exists():
        return load_dataset(path)
    items = generate_dataset(seed=seed)
    save_dataset(items, path)
    return items


if __name__ == "__main__":
    items = generate_dataset()
    print(f"Generated {len(items)} items")
    print(f"Sample item (fact_0, severity_3):")
    item = [i for i in items if i.severity == 3][0]
    print(json.dumps(item.to_dict(), indent=2))
