"""
Epistemic Memory Networks (EMN)
================================
Memory architecture treating confidence as a first-class architectural variable.

Built on NOVA's Dirichlet evidential uncertainty (EvidentialHead).
Three core components:
  1. EvidentialWriteGate    — assigns vacuity at memory write time
  2. EpistemicMemoryStore   — evicts by argmax(vacuity)
  3. UncertaintyWeightedRetriever — scores = cosine × (1 - vacuity)

Quick start:
    from emn.memory.store import EpistemicMemoryStore
    from emn.gates.write_gate import EvidentialWriteGate
    from emn.retrieval.retriever import UncertaintyWeightedRetriever
    from emn.llm.wrapped_lm import EMNWrappedCausalLM
"""

__version__ = "0.1.0"
__author__ = "Faaz Mohamed"
