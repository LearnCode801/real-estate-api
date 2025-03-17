# api/index.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import os

# Initialize FastAPI app
app = FastAPI(
    title="AdFrame Real Estate API",
    description="API for extracting house features from descriptions",
    version="1.0.0"
)

# Configure the Gemini API key
# In production, use environment variables
gemini_api_key = os.environ.get("GEMINI_API_KEY")

# Pydantic models for request and response
class HouseDescriptionRequest(BaseModel):
    description: str

class HouseFeatures(BaseModel):
    number_of_bedrooms: int = None
    number_of_bathrooms: int = None
    number_of_kitchens: int = None
    number_of_floors: str = None
    swimming_pool: str = None
    covered_area: str = None
    location: str = None
    price_of_the_house: str = None
    contact_number: str = None

# Define function to extract house features
def extract_house_features(text):
    template = """
    Extract the following details from the input text if present:
    - Number of bedrooms
    - Number of bathrooms
    - Number of kitchens
    - Number of floors
    - Swimming pool (yes/no)
    - Covered area
    - Location
    - Price of the house
    - Contact number
    Input text:
    {text}
    Strictly return valid JSON and Reference Response in the below format:
    
    "number_of_bedrooms": 2,
    "number_of_bathrooms": 2,
    "number_of_kitchens": 1,
    "number_of_floors": "no",
    "swimming_pool": "yes",
    "covered_area": "2500sqm",
    "location": "Attock",
    "price_of_the_house": "$1.5M",
    "contact_number": "0321-4090997"
    """
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=gemini_api_key)
    chain = LLMChain(prompt=prompt, llm=llm)
    result = chain.run(text)
    return result

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Hello I am Talha"}

# API endpoint
@app.get("/api")
async def api_root():
    return {"message": "Welcome to AdFrame Real Estate API. Use /api/extract-features endpoint to extract house features."}

# API endpoint to extract house features
@app.post("/api/extract-features", response_model=HouseFeatures)
async def extract_features(request: HouseDescriptionRequest):
    try:
        # Extract features from the description
        extracted_data = extract_house_features(request.description)
        
        # Process the extracted data
        extracted_data = str(extracted_data)
        
        # Parse the JSON data
        house_features = json.loads(extracted_data)
        
        return house_features
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
