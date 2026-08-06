from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    response_style: str = "balanced"
    language: str = "english"
    technical_level: str = "intermediate"


class UserStyle(BaseModel):
    preferred_answer_style: str = "balanced"
    likes_examples: bool = True
    likes_flowcharts: bool = False
    experience_level: str = ""


class UserProfile(BaseModel):
    user_id: str
    display_name: str = ""
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    interests: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    facts: List[str] = Field(default_factory=list)
    style: UserStyle = Field(default_factory=UserStyle)


class MemoryExtraction(BaseModel):
    preferences: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    facts: List[str] = Field(default_factory=list)
    style: Dict[str, Any] = Field(default_factory=dict)


class EpisodicMemoryHit(BaseModel):
    text: str
    memory_type: str = "summary"
    score: float = 0.0
    created_at: str = ""


class UserMemoryContext(BaseModel):
    profile: UserProfile
    episodic_memories: List[EpisodicMemoryHit] = Field(default_factory=list)
    frequent_topics: List[str] = Field(default_factory=list)
