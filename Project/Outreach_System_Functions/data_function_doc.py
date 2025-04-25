import os
import io
import json
import boto3
import re
from opensearchpy import OpenSearch
from docx import Document
from langchain_community.embeddings import BedrockEmbeddings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# === Global Config ===
s3_bucket = "project-outreach-system"
s3_prefix = "Project Case Study Outreach System/Data/Case Studies/"
index_name = "consulting-case-studies"
host = "https://search-academy-opensearch-02-qmvenlkzl2x7gaqcxhrq3onsdy.aos.eu-central-1.on.aws"
username = "academy-opensearch"
password = "8q%a^6uP@Yoqg71LIJEQVVhAu3lcYSOx#@Qs#w7E2IRJ3^!uIp"

s3 = boto3.client("s3")


# === Helper Functions ===
def get_bedrock_client():
    """
    Create and return a Bedrock Runtime client for the AWS region 'eu-central-1'.

    Returns:
        boto3.client: An AWS Bedrock Runtime client configured for the 'eu-central-1' region.
    """
    return boto3.client("bedrock-runtime", region_name="eu-central-1")


def identify_industry(text):
    """
    Identify the primary industry relevant to the given text.

    First attempts to extract the industry from a line explicitly stating it.
    If not found, uses an LLM via AWS Bedrock to determine the most relevant industry
    from a predefined list of candidates.

    Args:
        text (str): The input text from which to determine the industry.

    Returns:
        str: The identified industry label from the candidate list,
             or "Unknown" if it cannot be determined.
    """
    candidate_labels = [
        "Finance", "Healthcare", "Education", "Retail", "Technology", "Real Estate",
        "Telecommunications", "Manufacturing", "Transportation", "Energy",
        "Entertainment", "Legal", "Hospitality", "Agriculture", "Construction",
        "Government", "Aerospace", "Automotive", "Nonprofit", "Food & Beverage",
        "Insurance", "Sports", "Pharmaceuticals", "Mining", "Fashion"
    ]

    for line in text.splitlines():
        if "industry: " in line.lower():
            industries_from_text = line.split(":", 1)[1]
            industry_list = [i.strip().title() for i in industries_from_text.split(",")]
            for industry in industry_list:
                if industry in candidate_labels:
                    return industry

    prompt = f"""
    Human: Based on the following text, identify the primary industry it is most relevant to. 
    Only return one industry label from this list exactly: {', '.join(candidate_labels)}.
    If multiple apply, pick the most dominant one. 
    Text: {text}
    Assistant:"""

    try:
        bedrock = get_bedrock_client()
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0.2
        }
        response = bedrock.invoke_model(
            body=json.dumps(payload),
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response["body"].read())
        predicted = response_body["content"][0]["text"].strip()
        for label in candidate_labels:
            if label.lower() in predicted.lower():
                return label
        return "Unknown"
    except Exception as e:
        print(f"[ERROR] Industry identification failed: {e}")
        return "Unknown"

def summarize_solution_with_claude(text):
    """
    Summarize a case study using a structured format via Claude on AWS Bedrock.

    The summary includes:
    - **Challenge**: The problem addressed.
    - **Approach**: How the problem was solved.
    - **Impact**: The results or benefits achieved.

    Args:
        text (str): The full text of the case study to be summarized.

    Returns:
        str: A structured summary of the case study, or a fallback message if summarization fails.
    """
    bedrock = get_bedrock_client()
    prompt = f"""Human: Please summarize the following case study using the following structure:
    - **Challenge**: What problem was being addressed?
    - **Approach**: How was the problem approached or solved?
    - **Impact**: What results or benefits were achieved?

    Only include relevant information, and keep each section short and clear.

    Case Study:
    {text}

    Assistant:"""

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.7
    }

    try:
        response = bedrock.invoke_model(
            body=json.dumps(payload),
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            accept="application/json",
            contentType="application/json"
        )

        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    except Exception as e:
        print(f"[ERROR] Summary generation failed: {e}")
        return "Summary not available"


def extract_location_with_claude(text):
    """
    Extract the country in which a case study is based using Claude via AWS Bedrock.

    This function first looks for explicit location hints in the text, then uses a language model
    to infer the country based on both the full case study content and the extracted hints.

    Args:
        text (str): The full case study text, potentially including location hints.

    Returns:
        str: The name of the identified country (e.g., "Germany", "South Africa").
             Returns "Unknown" if the location cannot be confidently determined.
    """
    location_candidates = []
    for line in text.splitlines():
        if "location:" in line.lower():
            parts = line.split(":", 1)
            if len(parts) > 1:
                location_candidates.append(parts[1].strip())
    candidate_str = ", ".join(location_candidates) if location_candidates else "N/A"

    prompt = f"""
    Human: Please extract the country this case study is based in, using both the full case study text and any location hints like "{candidate_str}". 
    You do not need to construct a sentence as an answer to show your logic.
    Return only the country name (e.g., "Germany", "South Africa", "United States"). 
    If no location can be confidently identified, return "Unknown".

    Case Study:
    {text}

    Assistant:"""

    try:
        bedrock = get_bedrock_client()
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0
        }

        response = bedrock.invoke_model(
            body=json.dumps(payload),
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            accept="application/json",
            contentType="application/json"
        )

        response_body = json.loads(response['body'].read())
        return response_body["content"][0]["text"].strip()
    except Exception as e:
        print(f"[ERROR] Location extraction failed: {e}")
        return "Unknown"


def extract_casetudy_link(text):
    """
    Extract the first URL from the given text using a regular expression pattern.

    Args:
        text (str): The text potentially containing a URL.

    Returns:
        str: The first matched URL if found, otherwise "None".
    """
    url_pattern = r'https?://[^\s)]+' 
    matches = re.findall(url_pattern, text)
    return matches[0] if matches else "None"


def embed_text_titan(text_list):
    """
    Generate text embeddings for a list of documents using Amazon Titan via AWS Bedrock.

    This function attempts to embed each text in the input list. If an embedding cannot be
    generated for a particular entry, its index is added to a list of failures.

    Args:
        text_list (list of str): A list of text strings to embed.

    Returns:
        tuple:
            list: A list of embedding vectors (lists of floats) for successfully processed texts.
            list: A list of indices for texts that failed to embed.
    """
    bedrock = get_bedrock_client()
    embeddings = []
    failed_indices = []

    for i, text in enumerate(text_list):
        payload = { "inputText": text }
        try:
            response = bedrock.invoke_model(
                body=json.dumps(payload),
                modelId="amazon.titan-embed-text-v1",
                accept="application/json",
                contentType="application/json"
            )
            response_body = json.loads(response['body'].read())
            embedding = response_body.get("embedding")
            if not embedding:
                print(f"[WARN] No embedding returned for doc #{i}")
                failed_indices.append(i)
                continue
            embeddings.append(embedding)
        except Exception as e:
            print(f"[ERROR] Failed to embed doc #{i}: {e}")
            failed_indices.append(i)

    return embeddings, failed_indices


def main():
    """
    Main execution function to process case study documents from S3, extract metadata, 
    generate embeddings using Amazon Titan, and index them into an OpenSearch cluster.

    Workflow:
    1. Connects to OpenSearch using provided credentials.
    2. Lists all .docx files from a specified S3 bucket and prefix.
    3. Filters out documents already indexed in OpenSearch.
    4. For each new document:
        - Downloads and reads text content.
        - Identifies industry and location using Claude.
        - Extracts solution summary and source link.
        - Prepares metadata and stores documents locally.
    5. Generates vector embeddings for each document using Amazon Titan.
    6. Creates a new OpenSearch index if it doesn’t exist.
    7. Indexes each document into OpenSearch along with its embedding and metadata.

    Raises:
        Any exceptions are caught and printed during processing, embedding, and indexing.
    """
    documents, metadatas, ids = [], [], []
    client = OpenSearch(
        hosts=[host],
        http_auth=(username, password),
        use_ssl=True,
        verify_certs=False
    )

    response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
    files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.docx')]

    already_indexed_files = set()
    try:
        query_body = {
            "query": { "match_all": {} },
            "_source": ["metadata.file_name"],
            "size": 10000
        }
        response = client.search(index=index_name, body=query_body)
        for hit in response['hits']['hits']:
            file_name = hit['_source']['metadata']['file_name']
            already_indexed_files.add(file_name)
    except Exception as e:
        print(f"[ERROR] OpenSearch index fetch failed: {e}")

    new_files = [f for f in files if f.split('/')[-1] not in already_indexed_files]
    for i, s3_key in enumerate(new_files):
        print(f"\n📄 Processing S3 file: {s3_key}")
        file_obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        file_content = file_obj['Body'].read()
        doc = Document(io.BytesIO(file_content))
        full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip() != ""])

        industry = identify_industry(full_text)
        location = extract_location_with_claude(full_text)
        link = extract_casetudy_link(full_text)

        try:
            summary = summarize_solution_with_claude(full_text)
        except Exception as e:
            print(f"[ERROR] Summary failed for {s3_key}: {e}")
            continue

        metadata = {
            'file_name': s3_key.split('/')[-1],
            'industry': industry,
            'location': location,
            'solution_summary': summary,
            'casestudy_link': link,
            'doc_type': 'Case Study'
        }

        documents.append(full_text)
        metadatas.append(metadata)
        ids.append(str(i))

    embeddings, failed_indices = embed_text_titan(documents)
    documents = [doc for i, doc in enumerate(documents) if i not in failed_indices]
    metadatas = [meta for i, meta in enumerate(metadatas) if i not in failed_indices]
    ids = [id_ for i, id_ in enumerate(ids) if i not in failed_indices]

    if not client.indices.exists(index=index_name):
        vector_index_mapping = {
            "settings": { "index": { "knn": True } },
            "mappings": {
                "properties": {
                    "id": { "type": "keyword" },
                    "document": { "type": "text" },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 1536,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib"
                        }
                    },
                    "metadata": { "type": "object", "enabled": True }
                }
            }
        }
        client.indices.create(index=index_name, body=vector_index_mapping)
        print(f"✅ Index '{index_name}' created.")

    for i in range(len(documents)):
        body = {
            "id": ids[i],
            "document": documents[i],
            "embedding": embeddings[i],
            "metadata": metadatas[i]
        }
        client.index(index=index_name, id=ids[i], body=body)

    print(f"\nAll documents successfully indexed into OpenSearch!")

# === Entry Point ===
if __name__ == "__main__":
    main()
