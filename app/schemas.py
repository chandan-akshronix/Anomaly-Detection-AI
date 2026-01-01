from pydantic import BaseModel

class ProductInput(BaseModel):
    Count_Category: int
    Price_In_Dollar: float
    Length: float
    Width: float
    Height: float
    Final_Weights_in_Grams: float
    Hierarchy: str
