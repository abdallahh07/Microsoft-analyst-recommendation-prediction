from pydantic import BaseModel

class MicrosoftInput(BaseModel):
    Gross_Margin: float
    EBIT_Margin: float
    Net_Margin: float
    Revenue_Growth: float
    RD_Pct: float
    Current_Ratio: float
    Quick_Ratio: float
    Debt_To_Equity: float
    Debt_To_Assets: float
    ROE: float
    ROA: float
    Cash_Flow_Margin: float
    FCF_Margin: float
    GDP: float
    GDP_Growth: float
    FEDFUNDS: float
    FCF_Per_Share: float
    Intrinsic_Value: float
    Stock_Price: float
    Margin_Of_Safety: float

class MicrosoftOutput(BaseModel):
    prediction: str
    probability: float