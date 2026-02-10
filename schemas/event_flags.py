
from dataclasses import dataclass


@dataclass
class EventFlags:

    track_id: int

    weapon_score: float = 0.0
    fight_score: float = 0.0
    fallen_score: float = 0.0

    weapon: bool = False
    fight: bool = False
    fallen: bool = False
