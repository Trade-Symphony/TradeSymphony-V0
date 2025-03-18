from pydantic import BaseModel
from typing import List


class Industry(BaseModel):
    sector: str
    subIndustry: str


class ExpectedReturn(BaseModel):
    value: float
    timeframe: str


class RiskAssessment(BaseModel):
    level: str


class InvestmentThesis(BaseModel):
    recommendation: str
    conviction: str
    keyDrivers: List[str]
    expectedReturn: ExpectedReturn
    riskAssessment: RiskAssessment


class PositionSizingGuidance(BaseModel):
    allocationPercentage: float
    maximumDollarAmount: float
    minimumDollarAmount: float


class InvestmentRecommendationDetails(BaseModel):
    positionSizingGuidance: PositionSizingGuidance


class InvestmentRecommendation(BaseModel):
    name: str
    ticker: str
    industry: Industry
    investmentThesis: InvestmentThesis
    investmentRecommendationDetails: InvestmentRecommendationDetails


class InvestmentRecommendationList(BaseModel):
    recommendations: List[InvestmentRecommendation]
