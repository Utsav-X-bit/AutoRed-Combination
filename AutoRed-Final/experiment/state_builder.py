import json
import hashlib
import uuid
from typing import List, Dict, Any

class DefenseState:
    def __init__(
        self, 
        defense_type: str,
        access_type: str,
        difficulty: str,
        attempt: int,
        victim_behaviour: str,
        extractor_confidence: float,
        previous_strategies: List[str],
        remaining_strategies: List[str],
        local_memory: List[str]
    ):
        self.state_id = str(uuid.uuid4())
        self.defense_type = defense_type
        self.access_type = access_type
        self.difficulty = difficulty
        self.attempt = attempt
        self.victim_behaviour = victim_behaviour
        self.extractor_confidence = extractor_confidence
        self.previous_strategies = previous_strategies
        self.remaining_strategies = remaining_strategies
        self.local_memory = local_memory
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "defense_type": self.defense_type,
            "access_type": self.access_type,
            "difficulty": self.difficulty,
            "attempt": self.attempt,
            "victim_behaviour": self.victim_behaviour,
            "extractor_confidence": self.extractor_confidence,
            "previous_strategies": self.previous_strategies,
            "remaining_strategies": self.remaining_strategies,
            "local_memory": self.local_memory
        }

    def compute_hash(self) -> str:
        state_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(state_str.encode('utf-8')).hexdigest()

class StateBuilder:
    def __init__(self, all_strategies: List[str]):
        self.all_strategies = all_strategies

    def build_state(
        self,
        scenario: Any,
        attempt: int,
        previous_strategies: List[str],
        local_memory: List[str],
        last_victim_response: str = "",
        last_extractor_confidence: float = 0.0
    ) -> DefenseState:
        
        # Infer basic victim behaviour from response length/keywords
        victim_behaviour = "Neutral"
        if last_victim_response:
            resp_lower = last_victim_response.lower()
            if "i cannot" in resp_lower or "as an ai" in resp_lower:
                victim_behaviour = "Hard Refusal"
            elif len(last_victim_response) < 20:
                victim_behaviour = "Terse Refusal"
            else:
                victim_behaviour = "Partial Refusal / Engagement"

        remaining_strategies = [s for s in self.all_strategies if s not in previous_strategies]

        return DefenseState(
            defense_type=getattr(scenario, "defense_type", "Unknown"),
            access_type=getattr(scenario, "access_type", "Unknown"),
            difficulty=getattr(scenario, "difficulty", "Unknown"),
            attempt=attempt,
            victim_behaviour=victim_behaviour,
            extractor_confidence=last_extractor_confidence,
            previous_strategies=previous_strategies,
            remaining_strategies=remaining_strategies,
            local_memory=local_memory
        )
