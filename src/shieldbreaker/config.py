"""Method constants from Sections 4--7; no historical results are encoded here."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class AnalysisConfig:
    scales: tuple[int, ...] = (256, 512, 1024)
    stride: int = 512
    quantile: float = 0.995
    min_successes: int = 5
    min_wilson_lower: float = 0.70
    wilson_z: float = 1.959964

    def __post_init__(self):
        if not self.scales or any(w <= 0 for w in self.scales) or self.stride <= 0:
            raise ValueError("Scales and stride must be positive")
        if len(set(self.scales)) != len(self.scales):
            raise ValueError("Scales must be distinct")
        if not 0 < self.quantile < 1 or not 0 <= self.min_wilson_lower <= 1:
            raise ValueError("Invalid quantile or Wilson cutoff")
        if (
            self.min_successes < 0
            or not math.isfinite(self.wilson_z)
            or self.wilson_z <= 0
        ):
            raise ValueError("Invalid support configuration")

    def to_dict(self):
        result = asdict(self)
        result["scales"] = list(self.scales)
        return result


PRODUCTS = {
    "falcon": {
        "version": "7.15.18514.0",
        "processes": [
            "CSFalconService.exe",
            "CSFalconContainer.exe",
            "CsSystemTray_7.15.18514.0.exe",
        ],
    },
    "kaspersky": {
        "version": "21.3.10.391(b)",
        "database_date": "2025-05-04",
        "processes": ["avp.exe", "avpui.exe"],
    },
    "elastic": {
        "version": "8.12.2",
        "processes": ["elastic-agent.exe", "elastic-endpoint.exe"],
    },
}
# Names are configurable observation filters, not a claim of full sensor coverage.
