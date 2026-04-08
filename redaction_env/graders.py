import random
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DatasetItem:
    tokens: List[str]
    is_public_record: bool
    context_info: str
    ground_truth_mask: List[bool]

class GraderDataset:
    """
    Simulated dataset and grader serving examples across three difficulty tiers.
    In a production scenario, this might load from a JSON/HuggingFace dataset.
    """
    def __init__(self):
        self.easy_data = [
            DatasetItem(
                tokens=["John", "Doe", "'s", "SSN", "is", "123-456-7890", "."],
                is_public_record=False,
                context_info="Redact all basic PII (names, phone numbers, SSNs).",
                ground_truth_mask=[True, True, False, False, False, True, False]
            ),
            DatasetItem(
                tokens=["Contact", "Alice", "at", "555-0199", "for", "details", "."],
                is_public_record=False,
                context_info="Redact all basic PII (names, phone numbers, SSNs).",
                ground_truth_mask=[False, True, False, True, False, False, False]
            )
        ]
        
        self.medium_data = [
            DatasetItem(
                tokens=["The", "witness", ",", "Jane", "Smith", ",", "pointed", "at", "Judge", "Robert", "Brown", "."],
                is_public_record=True,
                context_info="Redact witness names. Do not redact court officials like judges.",
                ground_truth_mask=[False, False, False, True, True, False, False, False, False, False, False, False]
            ),
            DatasetItem(
                tokens=["Officer", "Johnson", "arrested", "the", "suspect", ",", "Michael", "."],
                is_public_record=True,
                context_info="Redact suspect names but keep police officer names visible.",
                ground_truth_mask=[False, False, False, False, False, False, True, False]
            )
        ]
        
        self.hard_data = [
            DatasetItem(
                tokens=["The", "person", "with", "the", "red", "mustache", "confessed", ".", "He", "said", "his", "name", "was", "Bob", "."],
                is_public_record=False,
                context_info="Redact the identity of the person with the red mustache based on context clues.",
                ground_truth_mask=[False, False, False, False, False, False, False, False, False, False, False, False, False, True, False]
            ),
            DatasetItem(
                tokens=["Client", "A", "transferred", "funds", "to", "their", "subsidiary", ".", "Client", "A", "is", "acme", "corp", "."],
                is_public_record=False,
                context_info="Redact the actual company name of Client A, but keep the pseudonym 'Client A'.",
                ground_truth_mask=[False, False, False, False, False, False, False, False, False, False, False, True, True, False]
            )
        ]

    def get_sample(self, difficulty: str = "easy") -> DatasetItem:
        """Fetch a random sample from the specified difficulty tier."""
        if difficulty == "easy":
            return random.choice(self.easy_data)
        elif difficulty == "medium":
            return random.choice(self.medium_data)
        elif difficulty == "hard":
            return random.choice(self.hard_data)
        else:
            raise ValueError("Invalid difficulty tier. Choose from 'easy', 'medium', 'hard'.")
