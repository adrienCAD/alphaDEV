"""Every stage writes a small JSON trust card that the site renders."""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class TrustCard:
    stage: str
    method: str
    inputs: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)  # physics-consistency checks
    notes: str = ""

    def write(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.stage}.trust.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path
