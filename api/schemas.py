"""Pydantic models for the Titanic prediction API."""
from pydantic import BaseModel, Field


class PassengerInput(BaseModel):
    pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    sex: str = Field(..., pattern="^(male|female)$", description="Sex of the passenger")
    age: float = Field(..., ge=0, le=100, description="Age in years")
    sibsp: int = Field(0, ge=0, le=8, description="Number of siblings/spouses aboard")
    parch: int = Field(0, ge=0, le=6, description="Number of parents/children aboard")
    fare: float = Field(..., ge=0, description="Ticket fare in GBP")
    embarked: str = Field("S", pattern="^[CSQ]$", description="Port of embarkation (C/S/Q)")

    model_config = {"json_schema_extra": {
        "examples": [{
            "pclass": 1, "sex": "female", "age": 29,
            "sibsp": 0, "parch": 0, "fare": 100.0, "embarked": "C",
        }]
    }}


class PredictionOutput(BaseModel):
    survived: bool
    probability: float = Field(..., ge=0, le=1)
    factors: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
