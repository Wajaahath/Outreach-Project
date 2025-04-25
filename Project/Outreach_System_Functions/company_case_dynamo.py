import pandas as pd
import re
import random
import requests
from bs4 import BeautifulSoup
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from langdetect import detect
import boto3
from boto3.dynamodb.conditions import Key, Attr


def extract_case_study_info(case_study_text):
    """
    Extract comprehensive information from case study text.

    :param case_study_text: str, full text of the case study
    :return: dict with extracted info: industry, location, solution details, tech keywords, and company size indicators
    """
    industry_match = re.search(r'Industry:\s*(.*?)(?:\n|$)', case_study_text)
    industry = industry_match.group(1).strip() if industry_match else ""

    location_match = re.search(r'Location:\s*(.*?)(?:\n|$)', case_study_text)
    location = location_match.group(1).strip() if location_match else ""

    solution_details = ""
    challenge_match = re.search(r'\*\*Challenge\*\*:\s*(.*?)(?:\*\*|\Z)', case_study_text, re.DOTALL)
    if challenge_match:
        solution_details += challenge_match.group(1).strip() + " "

    approach_match = re.search(r'\*\*Approach\*\*:\s*(.*?)(?:\*\*|\Z)', case_study_text, re.DOTALL)
    if approach_match:
        solution_details += approach_match.group(1).strip() + " "

    impact_match = re.search(r'\*\*Impact\*\*:\s*(.*?)(?:\*\*|\Z)', case_study_text, re.DOTALL)
    if impact_match:
        solution_details += impact_match.group(1).strip()

    tech_keywords = []
    tech_patterns = [
        r'data (?:and|&) analytics', r'cloud migration', r'data lakehouse',
        r'agile', r'predictive', r'forecasting', r'moderniz(?:e|ation)',
        r'integration', r'platform', r'digital transformation'
    ]
    for pattern in tech_patterns:
        if re.search(pattern, case_study_text, re.IGNORECASE):
            tech_keywords.append(re.search(pattern, case_study_text, re.IGNORECASE).group(0).lower())

    size_indicators = ["Fortune 500", "enterprise", "large", "mid-size", "small"]
    company_size = [indicator.lower() for indicator in size_indicators if re.search(indicator, case_study_text, re.IGNORECASE)]

    return {
        "industry": industry,
        "location": location,
        "solution_details": solution_details,
        "tech_keywords": tech_keywords,
        "company_size": company_size
    }


def extract_about_from_homepage(url):
    """
    Extracts an 'about'-like section from a company's homepage,
    only if the detected language is English.

    :param url: str, website URL
    :return: str or None
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        text_blocks = soup.find_all(["p", "div", "section"])
        about_keywords = ["we are", "our mission", "we believe", "founded", "company", "team", "who we are"]
        candidate_blocks = []
        for block in text_blocks:
            text = block.get_text(strip=True)
            if not text or len(text) < 100:
                continue
            try:
                if detect(text) != "en":
                    continue
            except:
                continue
            if any(kw in text.lower() for kw in about_keywords):
                candidate_blocks.append(text)
        if candidate_blocks:
            return max(candidate_blocks, key=len)
    except:
        pass
    return None


def scrape_company_website(company):
    """
    Scrapes a company website to extract about information.

    :param company: dict containing 'website' and 'name' keys
    :return: dict with 'website' and 'about_info'
    """
    website_url = company.get('website', '')
    company_name = company.get('name', 'Unknown company')
    if not website_url:
        return {"website": "", "about_info": "No website available"}
    if not website_url.startswith(('http://', 'https://')):
        website_url = 'https://' + website_url

    try:
        about_info = extract_about_from_homepage(website_url)
        if not about_info:
            about_info = extract_about_from_homepage(website_url.rstrip('/') + '/about')
            if not about_info:
                about_info = extract_about_from_homepage(website_url.rstrip('/') + '/about-us')
        return {
            "website": website_url,
            "about_info": about_info if about_info else "No about information found"
        }
    except Exception as e:
        return {"website": website_url, "about_info": f"Error scraping website: {str(e)}"}


def normalize_text(text):
    """
    Normalize text for better matching.

    :param text: str
    :return: str, lowercased and whitespace-normalized
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text.lower().strip())


def calculate_text_similarity(text1, text2):
    """
    Calculate similarity between two strings using SequenceMatcher.

    :param text1: str
    :param text2: str
    :return: float similarity ratio (0 to 1)
    """
    return SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio()


def parse_employee_count(size_str):
    """
    Convert size strings to numerical employee ranges.

    :param size_str: str, e.g. "1-10"
    :return: tuple (min, max)
    """
    size_mapping = {
        "1-10": (1, 10),
        "11-50": (11, 50),
        "51-200": (51, 200),
        "201-500": (201, 500),
        "501-1000": (501, 1000),
        "1K-5K": (1000, 5000),
        "5K-10K": (5000, 10000),
        "10K+": (10000, float('inf'))
    }
    return size_mapping.get(size_str, (0, float('inf')))


def map_case_study_size_to_company_size(case_size_indicators):
    """
    Map qualitative size indicators to employee count categories.

    :param case_size_indicators: list or str
    :return: list of size categories
    """
    size_indicators = normalize_text(' '.join(case_size_indicators) if isinstance(case_size_indicators, list) else case_size_indicators)
    if any(term in size_indicators for term in ["fortune 500", "fortune 100", "global 500"]):
        return ["1K-5K", "5K-10K", "10K+"]
    elif any(term in size_indicators for term in ["enterprise", "large enterprise"]):
        return ["501-1000", "1K-5K", "5K-10K", "10K+"]
    elif "large" in size_indicators:
        return ["201-500", "501-1000", "1K-5K"]
    elif any(term in size_indicators for term in ["mid-size", "medium", "mid-market"]):
        return ["51-200", "201-500"]
    elif any(term in size_indicators for term in ["small", "startup", "sme"]):
        return ["1-10", "11-50"]
    return ["1-10", "11-50", "51-200", "201-500", "501-1000", "1K-5K", "5K-10K", "10K+"]


def calculate_size_relevance(company_size, target_sizes):
    """
    Score company size relevance to case study indicators.

    :param company_size: str, e.g. "51-200"
    :param target_sizes: list of str
    :return: int score from 0 to 5
    """
    if not company_size or not target_sizes:
        return 0
    if company_size in target_sizes:
        return 5

    company_min, company_max = parse_employee_count(company_size)
    best_score = 0
    for target_size in target_sizes:
        target_min, target_max = parse_employee_count(target_size)
        if company_min <= target_max and company_max >= target_min:
            best_score = max(best_score, 4)
        else:
            distance = min(abs(target_min - company_max), abs(company_min - target_max))
            if distance <= 50:
                best_score = max(best_score, 3)
            elif distance <= 200:
                best_score = max(best_score, 2)
            elif distance <= 1000:
                best_score = max(best_score, 1)
    return best_score


def build_industry_map():
    """
    Construct a mapping of industry categories to relevant keywords.

    :return: dict mapping industries to keyword lists
    """
    industry_map = {
        "finance": {
            "keywords": ["banking", "investment", "financial services", "wealth management",
                         "capital markets", "fintech", "payments", "credit", "lending"]
        },
        # Extend as needed...
    }
    return industry_map
def calculate_industry_relevance(company_industry, case_study_industry):
    """
    Calculate relevance between a company's industry and a case study's industry.

    :param company_industry: Industry of the company
    :type company_industry: str
    :param case_study_industry: Industry mentioned in the case study
    :type case_study_industry: str
    :return: Relevance score between 0 and 5
    :rtype: int
    """
    if not company_industry or not case_study_industry:
        return 0

    company_industry = normalize_text(company_industry)
    case_study_industry = normalize_text(case_study_industry)

    if company_industry == case_study_industry:
        return 5

    industry_map = build_industry_map()
    max_score = 0

    for main_industry, mappings in industry_map.items():
        case_study_match = any(keyword in case_study_industry for keyword in [main_industry] + mappings["keywords"])
        company_match = any(keyword in company_industry for keyword in [main_industry] + mappings["keywords"])

        if case_study_match and company_match:
            max_score = max(max_score, 4)
            break

        if case_study_match and not company_match:
            for related in mappings.get("related", []):
                if related in company_industry:
                    max_score = max(max_score, 3)
                    break

    if max_score == 0:
        similarity = calculate_text_similarity(company_industry, case_study_industry)
        if similarity > 0.8:
            max_score = 4
        elif similarity > 0.6:
            max_score = 3
        elif similarity > 0.4:
            max_score = 2
        elif similarity > 0.2:
            max_score = 1

    return max_score


def calculate_tech_keywords_relevance(company_info, case_study_info):
    """
    Calculate relevance based on technical keywords and company type.

    :param company_info: Dictionary containing company information
    :type company_info: dict
    :param case_study_info: Dictionary containing case study information
    :type case_study_info: dict
    :return: Tech relevance score, capped at 10
    :rtype: int
    """
    relevance_score = 0

    solution_type_relevance = {
        "data analytics": {
            "Public Company": 5, "Privately Held": 4, "Self-Owned": 2, "Partnership": 3, "Government Agency": 4
        },
        "cloud migration": {
            "Public Company": 5, "Privately Held": 4, "Self-Owned": 2, "Partnership": 3, "Government Agency": 3
        },
        "digital transformation": {
            "Public Company": 5, "Privately Held": 4, "Self-Owned": 3, "Partnership": 3, "Government Agency": 4
        },
        "ai": {
            "Public Company": 5, "Privately Held": 4, "Self-Owned": 2, "Partnership": 3, "Government Agency": 3
        },
        "automation": {
            "Public Company": 4, "Privately Held": 5, "Self-Owned": 3, "Partnership": 3, "Government Agency": 4
        },
        "cyber security": {
            "Public Company": 5, "Privately Held": 4, "Self-Owned": 3, "Partnership": 3, "Government Agency": 5
        },
        "modernization": {
            "Public Company": 5, "Privately Held": 4, "Self-Owned": 2, "Partnership": 3, "Government Agency": 5
        }
    }

    company_type = company_info.get("type", "")

    for keyword in case_study_info.get("tech_keywords", []):
        keyword_lower = normalize_text(keyword)
        for solution_key, type_scores in solution_type_relevance.items():
            if solution_key in keyword_lower:
                relevance_score += type_scores.get(company_type, 1)
                break

    return min(relevance_score, 10)


def calculate_location_relevance(company_location, case_study_location):
    """
    Calculate location relevance based on country or region proximity.

    :param company_location: Location of the company
    :type company_location: str
    :param case_study_location: Location from the case study
    :type case_study_location: str
    :return: Location relevance score (0-3)
    :rtype: int
    """
    if not company_location or not case_study_location:
        return 0

    company_location = normalize_text(company_location)
    case_study_location = normalize_text(case_study_location)

    if company_location == case_study_location:
        return 3

    regions = {
        "north america": ["usa", "united states", "us", "canada", "mexico"],
        "europe": ["uk", "united kingdom", "germany", "france", "italy", "spain", "netherlands"],
        "asia pacific": ["china", "japan", "australia", "india", "singapore", "hong kong"],
        "latin america": ["brazil", "argentina", "chile", "colombia", "peru"],
        "middle east": ["uae", "dubai", "saudi arabia", "israel", "qatar", "oman"]
    }

    for region, countries in regions.items():
        company_in_region = region in company_location or any(country in company_location for country in countries)
        case_in_region = region in case_study_location or any(country in case_study_location for country in countries)

        if company_in_region and case_in_region:
            return 2

    return 0


def calculate_solution_relevance(company_info, case_study_info):
    """
    Calculate a comprehensive solution relevance score using multiple factors.

    :param company_info: Dictionary containing company details
    :type company_info: dict
    :param case_study_info: Dictionary containing case study details
    :type case_study_info: dict
    :return: Dictionary with total score and component scores
    :rtype: dict
    """
    if not company_info or not case_study_info:
        return 0

    scores = {}
    weights = {}

    weights["industry"] = 5
    scores["industry"] = calculate_industry_relevance(
        company_info.get("industry", ""),
        case_study_info.get("industry", "")
    )

    weights["size"] = 4
    scores["size"] = calculate_size_relevance(
        company_info.get("size", ""),
        map_case_study_size_to_company_size(case_study_info.get("company_size", []))
    )

    weights["tech"] = 3
    scores["tech"] = calculate_tech_keywords_relevance(company_info, case_study_info)

    weights["location"] = 2
    scores["location"] = calculate_location_relevance(
        company_info.get("country", ""),
        case_study_info.get("location", "")
    )

    total_weight = sum(weights.values())
    weighted_score = sum(scores[key] * weights[key] for key in scores.keys())
    max_possible_score = sum(5 * weights[key] for key in weights.keys())
    normalized_score = (weighted_score / max_possible_score) * 100

    detailed_scores = {
        "total_score": round(normalized_score, 2),
        "industry_score": scores["industry"],
        "size_score": scores["size"],
        "tech_score": scores["tech"],
        "location_score": scores["location"]
    }

    return detailed_scores
# def filter_and_rank_companies(df, case_study_info, min_score=30):
#     """Filter and rank companies with improved scoring and detailed results"""
#     if df.empty or not case_study_info:
#         return pd.DataFrame()
        
#     # Create lists to store results
#     all_scores = []
#     detailed_scores = []
    
#     # Calculate relevance score for each company
#     for index, company in df.iterrows():
#         company_info = company.to_dict()
#         score_details = calculate_solution_relevance(company_info, case_study_info)
        
#         total_score = score_details["total_score"]
#         if total_score >= min_score:  # Only include companies above threshold
#             all_scores.append((index, total_score))
#             detailed_scores.append((index, score_details))
    
#     # Sort companies by total relevance score (descending)
#     all_scores.sort(key=lambda x: x[1], reverse=True)
    
#     # Get the top companies
#     if all_scores:
#         top_indices = [score[0] for score in all_scores]
#         matching_companies = df.loc[top_indices].copy()
        
#         # Add detailed scores to the dataframe
#         for score_detail in detailed_scores:
#             idx = score_detail[0]
#             details = score_detail[1]
            
#             for key, value in details.items():
#                 matching_companies.loc[idx, key] = value
        
#         # Sort by total score
#         matching_companies = matching_companies.sort_values('total_score', ascending=False)
        
#         return matching_companies
    
#     return pd.DataFrame()  # Return empty dataframe if no matches


def filter_companies_from_dynamodb(table, case_study_info, min_score=30):
    """
    Query and filter companies from a DynamoDB table based on a case study and relevance scoring.

    This function scans a DynamoDB table to retrieve company profiles and filters them
    based on their relevance to the provided case study using multiple dimensions 
    (industry, size, tech, and location). Relevance is computed using a composite score.
    Optionally applies an initial filter on industry before falling back to a full table scan.

    :param table: DynamoDB table resource (boto3 Table)
    :type table: boto3.resources.factory.dynamodb.Table
    :param case_study_info: Information about the case study used for scoring relevance
    :type case_study_info: dict
    :param min_score: Minimum total relevance score for a company to be included
    :type min_score: int
    :return: List of company dicts with total and individual relevance scores, sorted by total score
    :rtype: list[dict]
    """
    matching_companies = []

    # Build filter expression for initial filtering if possible
    filter_expressions = []

    # Try to filter by industry first using FilterExpression
    if case_study_info.get("industry"):
        try:
            # First try an exact industry match
            response = table.scan(
                FilterExpression=Attr("industry").eq(case_study_info["industry"])
            )
            items = response["Items"]

            # Add paginated results
            while "LastEvaluatedKey" in response:
                response = table.scan(
                    FilterExpression=Attr("industry").eq(case_study_info["industry"]),
                    ExclusiveStartKey=response["LastEvaluatedKey"]
                )
                items.extend(response["Items"])

            if len(items) < 10:
                do_broader_scan = True
            else:
                do_broader_scan = False

        except Exception:
            do_broader_scan = True
    else:
        do_broader_scan = True

    if do_broader_scan:
        # Scan entire table with pagination
        items = []
        response = table.scan()
        items.extend(response["Items"])

        while "LastEvaluatedKey" in response:
            response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            items.extend(response["Items"])

    # Process and score each item
    for item in items:
        score_details = calculate_solution_relevance(item, case_study_info)
        total_score = score_details["total_score"]

        if total_score >= min_score:
            for key, value in score_details.items():
                item[key] = value
            matching_companies.append(item)

    matching_companies.sort(key=lambda x: x["total_score"], reverse=True)

    return matching_companies

def select_target_companies(matching_companies, min_score=40):
    """
    Select and return companies that exceed a specified score threshold.

    This function filters out companies from a list of matching candidates based
    on a higher score threshold for use in targeted actions (e.g., outreach, partnership).
    Falls back to returning all matching companies if none meet the minimum threshold.

    :param matching_companies: List of companies with scored relevance
    :type matching_companies: list[dict]
    :param min_score: Minimum score threshold for inclusion in final selection
    :type min_score: int
    :return: Filtered list of companies with scores >= min_score
    :rtype: list[dict]
    """
    if not matching_companies:
        return []

    qualified_companies = [company for company in matching_companies if company["total_score"] >= min_score]

    if not qualified_companies:
        return matching_companies

    return qualified_companies

# def select_target_companies(matching_companies, min_score=40): 
#     """Select target companies with minimum score threshold (no result limit)"""
#     if matching_companies.empty:
#         return None

#     # Filter by minimum score
#     qualified_companies = matching_companies[matching_companies['total_score'] >= min_score]

#     if qualified_companies.empty:
#         # Fall back to original companies if none meet threshold
#         return matching_companies

#     # Return all qualified companies (no .head())
#     return qualified_companies
#"main" function that calls all the other functions 
def process_case_study(case_study_text):
    """
    Process a case study text, retrieve and filter relevant companies, 
    and find a target company with available about information.

    This function extracts information from a case study, queries DynamoDB for 
    matching companies, and processes each company to find one with retrievable 
    information about the company. The results are logged, and the details of 
    the selected company are returned.

    :param case_study_text: The text of the case study to process
    :type case_study_text: str
    :return: A tuple containing:
        - A list of strings representing the logged output during the processing
        - A list of company details (name, industry, size, country, website)
    :rtype: tuple[list[str], list[str]]
    :raises Exception: If an error occurs during processing (e.g., network failure, DynamoDB issue)
    """
    results_log = []

    # Extract case study information
    case_study_info = extract_case_study_info(case_study_text)
    results_log.append("\nCase Study Information:")
    for key, value in case_study_info.items():
        line = f"{key}: {value}"
        results_log.append(line)
    
    try:
        # Initialize DynamoDB connection
        dynamodb = boto3.resource('dynamodb', region_name='eu-central-1')  
        table = dynamodb.Table('Targeted_companies')  # DynamoDB table name
        
        # Query and filter companies directly from DynamoDB
        matching_companies = filter_companies_from_dynamodb(table, case_study_info)
        msg = f"\nFound {len(matching_companies)} matching companies from DynamoDB"
        results_log.append(msg)

        if not matching_companies:
            msg = "No suitable companies found."
            results_log.append(msg)
            return results_log

        # Select the target companies
        target_companies = select_target_companies(matching_companies)
        selected_company = None
        selected_company_info = None

        msg = "\nSearching for target company with available about information..."
        results_log.append(msg)

        top_n = 50
        # Using slicing for list instead of DataFrame's head() method
        top_target_companies = target_companies[:top_n]

        msg = f"\nTop {top_n} Target Companies:"
        results_log.append(msg)

        # Create a table-like display for the top companies
        table_display = []
        header = "| Name | Industry | Size | Country | Website | Total Score |"
        separator = "|------|----------|------|---------|---------|-------------|"
        table_display.append(header)
        table_display.append(separator)
        
        for company in top_target_companies:
            row = f"| {company.get('name', 'N/A')} | {company.get('industry', 'N/A')} | {company.get('size', 'N/A')} | {company.get('country', 'N/A')} | {company.get('website', 'N/A')} | {company.get('total_score', 0)} |"
            table_display.append(row)
        
        for row in table_display:
            results_log.append(row)

        # Iterating through a list of dictionaries instead of DataFrame rows
        options = top_target_companies
        for i in range(len(target_companies)):
            random_number = random.randint(0, len(options) - 1)
            company = options[random_number]

            msg = f"\nProcessing {company.get('name', '')} (Record #{random_number + 1})..."
            results_log.append(msg)

            company_info = scrape_company_website(company)
            about_info = company_info.get('about_info', '')

            if (about_info and 
                about_info != "No about information found" and 
                not about_info.startswith("Error scraping website:") and
                not about_info == "No website available"):

                selected_company = company
                selected_company_info = company_info

                msg = f"✓ Successfully found about information in English for {company.get('name', '')}"
                results_log.append(msg)
                break
            else:
                msg = f"✗ Could not retrieve valid English about information for {company.get('name', '')}"
                results_log.append(msg)
                options.pop(random_number) 

        # Prepare the results for the selected company
        company_details = []

        if selected_company is not None:
            results_log.append("\n" + "=" * 50)
            results_log.append(f"FINAL SELECTED COMPANY: {selected_company.get('name', '')}")
            results_log.append("=" * 50)
            results_log.append(f"Selected Company Name: {selected_company.get('name', '')}")
            results_log.append(f"Industry: {selected_company.get('industry', 'N/A')}")
            results_log.append(f"Size: {selected_company.get('size', 'N/A')}")
            results_log.append(f"Country: {selected_company.get('country', 'N/A')}")
            results_log.append(f"Website: {selected_company_info.get('website', 'N/A')}")
            results_log.append(f"Relevance Score: {selected_company.get('total_score', 0)}")
            results_log.append("\nAbout Information:")
            results_log.append(selected_company_info.get('about_info', ''))

            company_details.append(f"Name: {selected_company.get('name', '')}")
            company_details.append(f"Industry: {selected_company.get('industry', '')}")
            company_details.append(f"Size: {selected_company.get('size', '')}")
            company_details.append(f"Country: {selected_company.get('country', '')}")
            company_details.append(f"Website: {selected_company.get('website', '')}")
            
            return results_log, company_details
        else:
            msg = "\nNone of the top companies had retrievable about information."
            results_log.append(msg)
            return results_log

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        results_log.append(error_msg)
        import traceback
        traceback_msg = traceback.format_exc()
        results_log.append(traceback_msg)
        return results_log

if __name__ == "__main__":
    company_match_info = process_case_study()
