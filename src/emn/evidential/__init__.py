"""EMN Evidential module — NOVA bridge and uncertainty head."""

from emn.evidential.nova_uncertainty import EvidentialHead, ConformalCalibrator, UncertaintyEstimationModule
from emn.evidential.nova_bridge import VacuityExtractor

__all__ = [
    "EvidentialHead",
    "ConformalCalibrator",
    "UncertaintyEstimationModule",
    "VacuityExtractor",
]
