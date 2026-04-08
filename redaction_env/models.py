# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Redaction Env Environment.

The redaction_env environment is a simple test environment that echoes back messages.
"""

import json
from openenv.core.env_server.types import Action, Observation
from pydantic import Field, field_validator


class RedactionAction(Action):
    """Action for the Redaction Env environment - a boolean mask for token redaction."""

    redact_mask: list[bool] = Field(
        ..., 
        description="Boolean mask indicating which tokens to redact (True = redact, False = keep)"
    )

    @field_validator('redact_mask', mode='before')
    @classmethod
    def parse_mask(cls, v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v_stripped = v.strip()
            
            # 1. Try parsing strictly as JSON first
            if v_stripped.startswith('['):
                # JSON requires lowercase boolean "true" and "false"
                # If they typed Python style "True" or "False", let's fix it for the json parser
                v_json_ready = v_stripped.replace('True', 'true').replace('False', 'false')
                try:
                    return json.loads(v_json_ready)
                except ValueError:
                    pass
            
            # 2. Fallback to space/comma separated parsing
            # Replace commas and brackets so we can split by spaces safely
            cleaned = v_stripped.replace('[', ' ').replace(']', ' ').replace(',', ' ')
            tokens = cleaned.split()
            
            mask = []
            for t in tokens:
                lower_t = t.lower()
                if lower_t in ('true', '1', 't'):
                    mask.append(True)
                elif lower_t in ('false', '0', 'f'):
                    mask.append(False)
                else:
                    raise ValueError(f"Cannot parse boolean from '{t}'")
            return mask
            
        raise ValueError(f"Invalid input format for redact_mask: {type(v)}")


class RedactionObservation(Observation):
    """Observation from the Redaction Env environment - the tokenized text and context."""

    tokens: list[str] = Field(
        default_factory=list, 
        description="List of text tokens representing the input to be redacted"
    )
    is_public_record: bool = Field(
        default=False, 
        description="Flag indicating if the text is from a public record (affects redaction strictness)"
    )
    context_info: str = Field(
        default="", 
        description="Additional context or role information (e.g., 'redact witness names but not judge names')"
    )
