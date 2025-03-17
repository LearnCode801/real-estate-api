# api/index.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
# from langchain.chains import create_structured_output_chain
import json
import os

# Initialize FastAPI app
app = FastAPI(
    title="AdFrame Real Estate API",
    description="API for extracting house features from descriptions",
    version="1.0.0"
)

# Configure the Gemini API key
gemini_api_key = os.environ.get("GEMINI_API_KEY" )

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
    prompt_template = """
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
    """
    
    # Create the prompt
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["text"]
    )
    
    # Define the output schema to match our expected format
    schema = {
        "properties": {
            "number_of_bedrooms": {"type": "integer"},
            "number_of_bathrooms": {"type": "integer"},
            "number_of_kitchens": {"type": "integer"},
            "number_of_floors": {"type": "string"},
            "swimming_pool": {"type": "string"},
            "covered_area": {"type": "string"},
            "location": {"type": "string"},
            "price_of_the_house": {"type": "string"},
            "contact_number": {"type": "string"}
        },
        "required": []
    }
    
    # Initialize the language model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key)
    
    # Use a simpler approach with just the LLM and a system prompt
    response = llm.invoke(
        f"""
        {prompt_template.replace("{text}", text)}
        
        Return ONLY a valid JSON object with the following format:
        {{
          "number_of_bedrooms": 2,
          "number_of_bathrooms": 2,
          "number_of_kitchens": 1,
          "number_of_floors": "no",
          "swimming_pool": "yes",
          "covered_area": "2500sqm",
          "location": "Attock",
          "price_of_the_house": "$1.5M",
          "contact_number": "0321-4090997"
        }}
        """
    )
    
    # Extract JSON from the response
    content = response.content
    
    # Sometimes the model returns the JSON with markdown code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    
    # Parse the JSON data
    house_features = json.loads(content)
    return house_features

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
        house_features = extract_house_features(request.description)
        return house_features
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
