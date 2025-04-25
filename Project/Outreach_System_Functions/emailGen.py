####### IMPORTS #########################################################
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

### Structure model output
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

########## HELPER FUNCTIONS ##############################################
def animatePrint(text, speed=70):
    """
    Print text with a typewriter-style animation to the terminal.

    This function prints each character of the text with a delay between them, preventing text dumps 
    in the terminal and aiding in readability when displaying content, especially for longer text.

    :param text: The text to animate.
    :type text: str
    :param speed: Characters per second, defaults to 70.
    :type speed: int, optional
    """
    delay = 1 / speed
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()

######### CONFIGURE MODEL #################################################
# Set up email generation model
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
model = ChatBedrock(model_id=model_id)

# Structured model
class Email(BaseModel):
    to: str = Field("Who the email is being sent to: A dummy email address using the target company name as the domain")
    subject: str = Field(description="The subject of the email. It should contain your company name (Elixirr). The subject should be something you'd want to click on")
    body: str = Field(description="The body of the email. Structured in markdown format")

structuredModel = model.with_structured_output(Email)

# Define the prompt
system_template = (
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
    "- Include a link to the case study: [use a dummy link for now].\n"
    "- Let them know they can reach out with questions or if they want to chat.\n"
    "- If helpful for clarity, use bullet points or numbering.\n\n"
    "- Propose a meeting:\n"
    "- If the company is in the UK, USA, or South Africa, suggest an in-person meeting.\n"
    "- Otherwise, propose a Microsoft Teams meeting and include a placeholder link to schedule.\n\n"

    "Structure the email as follows in markdown (do not include template labels):\n"
    "1. Greeting\n"
    "2. Hook: Brief, interesting opening\n"
    "3. Relevance: Tie a specific company insight to Elixirr's capability\n"
    "4. Value: Say how Elixirr can help (light case study reference)\n"
    "5. Invite them to ask questions or reach out\n"
    "6. Meeting proposal with relevent links"
    "7. Friendly sign-off mentioning Elixirr\n\n"

    "---- END OF SYSTEM INSTRUCTIONS ----"
)

############# GENERATING EMAIL TEMPLATE EXAMPLE #########################################################################################
def generateEmail(caseStudyDetails, targetCompanyDetails, verbose=True):
    """
    Generates an outreach email using structured AI output based on a provided case study and target company details.

    :param caseStudyDetails: Case study text describing Elixirr's previous engagement.
    :type caseStudyDetails: str
    :param targetCompanyDetails: Insights or notes about the target company.
    :type targetCompanyDetails: str
    :param verbose: If True, animates the output.
    :type verbose: bool
    :return: A structured email object.
    :rtype: Email
    """
    if verbose: print("Generating email...")

    human = (
        "With reference to the following case study:\n"
        f"{caseStudyDetails}\n\n"
        "Write an outreach email to the following company. Here are the details of the target company:\n"
        f"{targetCompanyDetails}"
    )

    email = structuredModel.invoke([
        SystemMessage(content=system_template),
        HumanMessage(content=(human)),
    ])

    if verbose:
        animatePrint("########## EMAIL TEMPLATE #################")
        animatePrint("\nTo: " + email.to)
        animatePrint("\nSubject: " + email.subject)
        animatePrint("\nBody:\n" + email.body + "\n")

    return email

def tweakEmail(email, changes, verbose=True):
    """
    Applies modifications to an already generated email while adhering to email writing guidelines.

    :param email: The original email object.
    :type email: Email
    :param changes: Textual description of the desired changes.
    :type changes: str
    :param verbose: If True, animates the modified output.
    :type verbose: bool
    :return: A modified Email object.
    :rtype: Email
    """
    if verbose:
        print("Modifying email. Changes:", changes)

    newEmail = structuredModel.invoke([
        SystemMessage(content="Given the following pydantic email object\n"
            f"Subject: {email.subject}\n"
            f"Body: {email.body}\n\n"

            "Follow these email writing guidelines:\n"
            "- Keep it short: 50-100 words max.\n"
            "- Sound human. Keep it light, low-pressure, and professional—humour is welcome if it fits.\n"
            "- Don't push for a meeting or assume interest. Your aim is to spark curiosity.\n"
            "- Avoid generic marketing phrases (e.g. 'proven track record').\n"
            "- Focus on something specific about the target company and how Elixirr could help.\n"
            "- Mention the case study briefly, only as relevant background.\n"
            "- Include a link to the case study: [use a dummy link for now].\n"
            "- Let them know they can reach out with questions or if they want to chat.\n"
            "- If helpful for clarity, use bullet points or numbering.\n\n"
            "- Propose a meeting:\n"
            "- If the company is in the UK, USA, or South Africa, suggest an in-person meeting.\n"
            "- Otherwise, propose a Microsoft Teams meeting and include a placeholder link to schedule.\n\n"

            "Structure the email as follows in markdown (do not include template labels):\n"
            "1. Greeting\n"
            "2. Hook: Brief, interesting opening\n"
            "3. Relevance: Tie a specific company insight to Elixirr's capability\n"
            "4. Value: Say how Elixirr can help (light case study reference)\n"
            "5. Invite them to ask questions or reach out\n"
            "6. Meeting proposal with relevent links"
            "7. Friendly sign-off mentioning Elixirr\n\n"

            "and apply the following changes to the original email:"),
        HumanMessage(content=changes),
    ])

    if verbose:
        animatePrint("########## MODIFIED EMAIL #################")
        animatePrint("\nTo: " + newEmail.subject)
        animatePrint("\nSubject: " + newEmail.subject)
        animatePrint("\nBody:\n" + newEmail.body + "\n")

    return newEmail