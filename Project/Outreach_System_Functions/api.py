import json
import boto3
import time
import urllib3
from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from opensearchpy import OpenSearch
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from api_python_stuff.company_case_dynamo import process_case_study
from api_python_stuff.emailGen import generateEmail, tweakEmail
from api_python_stuff.test import extractCompanyDetails
import logging
import asyncio
from typing import Optional
# --------------------------------------ENDPOINT SETUP-----------------------
# Disable warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# FastAPI setup
app = FastAPI(title="Case Studies API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenSearch config
host = "https://search-academy-opensearch-02-qmvenlkzl2x7gaqcxhrq3onsdy.aos.eu-central-1.on.aws"
username = "academy-opensearch"
password = "8q%a^6uP@Yoqg71LIJEQVVhAu3lcYSOx#@Qs#w7E2IRJ3^!uIp"
index_name = "consulting-case-studies"

# S3 Config (not used here, retained)
s3_bucket = "project-outreach-system"
s3_prefix = "Project Case Study Outreach System/Data/Case Studies/"

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# OpenSearch client
def get_opensearch_client():
    """
    Create and return an OpenSearch client instance.

    :return: OpenSearch client
    :rtype: OpenSearch
    """
    return OpenSearch(
        hosts=[host],
        http_auth=(username, password),
        use_ssl=True,
        verify_certs=False
    )

# Bedrock client
def get_bedrock_client():
    """
    Create and return a Bedrock runtime client.

    :return: Bedrock runtime client
    :rtype: botocore.client.BaseClient
    """
    return boto3.client("bedrock-runtime", region_name="eu-central-1")

# Embedding function (async)
async def embed_text_titan(text_list):
    """
    Generate Amazon Titan embeddings for a list of input texts.

    :param text_list: List of input strings to embed
    :type text_list: list[str]
    :return: Tuple of embeddings and failed indices
    :rtype: tuple[list[list[float]], list[int]]
    """
    bedrock = get_bedrock_client()
    embeddings, failed = [], []
    for i, text in enumerate(text_list):
        try:
            response = bedrock.invoke_model(
                body=json.dumps({ "inputText": text }),
                modelId="amazon.titan-embed-text-v1",
                accept="application/json",
                contentType="application/json"
            )
            embedding = json.loads(response['body'].read()).get("embedding")
            if embedding:
                embeddings.append(embedding)
            else:
                failed.append(i)
        except Exception as e:
            logger.error(f"Error generating embedding for text index {i}: {e}")
            failed.append(i)
    return embeddings, failed

# ----------------- CASE STUDY ENDPOINTS -----------------

@app.get("/")
async def root():
    """
    Root endpoint to confirm the API is running.

    :return: Welcome message
    :rtype: dict
    """
    return {"message": "Case Studies API is running"}

@app.get("/metadata")
async def get_metadata():
    """
    Retrieve all metadata from the OpenSearch index.

    :return: List of metadata dictionaries
    :rtype: list[dict]
    """
    client = get_opensearch_client()
    response = client.search(index=index_name, body={"query": {"match_all": {}}, "_source": ["metadata"], "size": 1000})
    return [hit["_source"]["metadata"] for hit in response["hits"]["hits"] if "metadata" in hit["_source"]]

@app.get("/search")
async def search_case_studies(query: str, size: int = 3):
    """
    Search case studies using a semantic vector-based query.

    :param query: The search query string
    :type query: str
    :param size: Number of search results to return
    :type size: int
    :return: List of matching case study metadata and scores
    :rtype: list[dict]
    """
    client = get_opensearch_client()
    query_embedding, _ = await embed_text_titan([query])
    if not query_embedding:
        raise HTTPException(status_code=500, detail="Failed to generate embedding")
    
    search_query = {
        "size": size,
        "query": {
            "knn": {
                "embedding": {"vector": query_embedding[0], "k": size}
            }
        }
    }
    response = client.search(index=index_name, body=search_query, size=size)
    return [{
        "file_name": hit['_source']['metadata']['file_name'],
        "industry": hit['_source']['metadata']['industry'],
        "location": hit['_source']['metadata']['location'],
        "solution_summary": hit['_source']['metadata']['solution_summary'],
        "casestudy_link": hit['_source']['metadata']['casestudy_link'],
        "doc_type": hit['_source']['metadata']['doc_type'],
        "score": hit['_score']
    } for hit in response['hits']['hits']]

@app.get("/industries")
async def get_industries():
    """
    Get all unique industries from the case study metadata.

    :return: List of industries
    :rtype: list[str]
    """
    client = get_opensearch_client()
    response = client.search(index=index_name, body={"query": {"match_all": {}}, "_source": ["metadata.industry"], "size": 1000})
    return list({hit["_source"]["metadata"]["industry"] for hit in response["hits"]["hits"] if "metadata" in hit["_source"] and "industry" in hit["_source"]["metadata"] and hit["_source"]["metadata"]["industry"] != "Unknown"})

@app.get("/locations")
async def get_locations():
    """
    Get all unique locations from the case study metadata.

    :return: List of locations
    :rtype: list[str]
    """
    client = get_opensearch_client()
    response = client.search(index=index_name, body={"query": {"match_all": {}}, "_source": ["metadata.location"], "size": 1000})
    return list({hit["_source"]["metadata"]["location"] for hit in response["hits"]["hits"] if "metadata" in hit["_source"] and "location" in hit["_source"]["metadata"] and hit["_source"]["metadata"]["location"] != "Unknown"})

# ----------------- EMAIL GENERATION ENDPOINT -----------------

# Email model
class Email(BaseModel):
    body: str = Field(description="The body of the email")

class EmailRequest(BaseModel):
    file_name: str
    summary: str

model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
model = ChatBedrock(model_id=model_id)
structured_model = model.with_structured_output(Email)

system_prompt = (
    "-- SYSTEM INSTRUCTIONS --\n"
    "You are a helpful assistant tasked with drafting short outreach emails for potential clients of the company 'Elixirr'.\n"
    "Your goal is to start a light, low-pressure conversation by showing how Elixirr's solutions (as referenced in the case study below) align with a challenge or opportunity at the target company.\n\n"
    "Using the case study and company details provided, write a concise, friendly, and relevant email that feels human—not salesy.\n\n"
    "Return the output in **Markdown format**.\n\n"
    "Guidelines for writing the email:\n"
    "Follow these email writing guidelines:\n"
    "- Keep it short: 50-100 words max.\n"
    "- Sound human. Keep it light, low-pressure, and professional—humour is welcome if it fits.\n"
    "- Don't push for a meeting or assume interest. Your aim is to spark curiosity.\n"
    "- Avoid generic marketing phrases (e.g. 'proven track record').\n"
    "- Focus on something specific about the target company and how Elixirr could help.\n"
    "- Mention the case study briefly, only as relevant background.\n"
    "- Include a link to the case study. If there is no link available then skip this step.\n"
    "- Let them know they can reach out with questions or if they want to chat.\n"
    "- If helpful for clarity, use bullet points or numbering.\n\n"
    "- Propose a meeting:\n"
    "- If the company is in the UK, USA, or South Africa, suggest an in-person meeting.\n"
    "- Otherwise, propose a Microsoft Teams meeting and include a placeholder link to schedule.\n\n"
    "- Include the link to the case study in the email.\n\n"
    
    "Structure the email as follows in markdown (do not include template labels):\n"
    "1. Greeting (DON'T SAY HI THERE...  Dear is acceptable)\n"
    "2. Hook: Brief, interesting opening\n"
    "3. Relevance: Tie a specific company insight to Elixirr's capability\n"
    "4. Value: Say how Elixirr can help (light case study reference)\n"
    "5. Invite them to ask questions or reach out\n"
    "6. Meeting proposal with relevent links"
    "7. Friendly sign-off mentioning Elixirr\n\n"
    "---- END OF SYSTEM INSTRUCTIONS ----"
)

class CaseStudyInput(BaseModel):
    case_study_text: str

class EmailGenerationRequest(BaseModel):
    file_name: str
    summary: str
    industry: str
    location: str
    link: str
    additional_info: str = ""
    
def parse_ui_data(ui_data_list):
    """
    Parse a list of UI data strings into a structured dictionary.

    :param ui_data_list: List of strings in 'key: value' format
    :type ui_data_list: list[str]
    :return: Parsed company details
    :rtype: dict
    """
    details = {}
    for item in ui_data_list:
        if ':' in item:
            key, value = item.split(':', 1)
            details[key.strip().lower()] = value.strip()
    return {
        "name": details.get("name", ""),
        "industry": details.get("industry", ""),
        "size": details.get("size", ""),
        "country": details.get("country", ""),
        "website": details.get("website", "")
    }

@app.post("/generate-email")
async def generate_email(data: EmailGenerationRequest):
    """
    Generate a marketing outreach email based on case study metadata.

    :param data: Email generation request with case study info
    :type data: EmailGenerationRequest
    :return: Dictionary containing the generated email and company name
    :rtype: dict
    """
    try:
        # Prepare case study text
        case_study_text = f"""
        Title: {data.file_name}
        Industry: {data.industry}
        Location: {data.location}
        Link: {data.link}
        Summary: {data.summary}
        Additional Info: {data.additional_info}
        """
        
        # Process using existing functions
        targetDetails, uiData = process_case_study(case_study_text)
        details = extractCompanyDetails("\n".join(targetDetails[-11:]))

        # Don't use await with invoke()
        email = structured_model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Case study: {case_study_text}\nCompany Details: {details}")
        ])
        
        company_details = parse_ui_data(uiData)
        print("UI DATA:", company_details)
        company_name = company_details.get("name", "Company")
        print("Company Name:", company_name)
        return {
            "email": email.body,
            "company_name": company_name,
        }
    except Exception as e:
        logger.error(f"Error generating email: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate email")
    

# -------------------- EMAIL ALTERING ------------------------

class EmailChangeRequest(BaseModel):
    email: str  # Just the email body as a string
    changes: str  # The textual description of changes to apply

@app.post("/tweak-email")
async def tweak_email(request: EmailChangeRequest):
    """
    Modify an existing email based on textual instructions.

    :param request: Request containing the original email and changes
    :type request: EmailChangeRequest
    :return: Modified email body
    :rtype: dict
    """
    try:
        changes = request.changes
        email = request.email
        
        modified_email = tweakEmail( email, changes)
        return {"modified_email": modified_email.model_dump()}
    except Exception as e:
        logger.error(f"Error tweaking email: {e}")
        raise HTTPException(status_code=500, detail=f"Error tweaking email: {str(e)}")
    
# --------------------------------------------------------------------------------------------------

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
