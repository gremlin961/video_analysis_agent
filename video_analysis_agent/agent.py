# Copyright 2024 Google, LLC. This software is provided as-is,
# without warranty or representation for any use or purpose. Your
# use of it is subject to your agreement with Google.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Example Agent Workflow using Google's ADK
# 
# This notebook provides an example of building an agentic workflow with Google's new ADK. 
# For more information please visit  https://google.github.io/adk-docs/



# Vertex AI Modules
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part 

# Vertex Agent Modules
from google.adk.agents.llm_agent import Agent # Base class for creating agents
from google.adk.tools.agent_tool import AgentTool # Wrapper to use one agent as a tool for another
from google.adk.tools import ToolContext
from google.adk.tools import load_artifacts, load_memory

# Import the ADK BigQuery tool
from google.adk.tools.bigquery import BigQueryCredentialsConfig
from google.adk.tools.bigquery import BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig
from google.adk.tools.bigquery.config import WriteMode

# Imports for Agent Engine
from vertexai import agent_engines

import google.auth

# Vertex GenAI Modules (Alternative/Legacy way to interact with Gemini, used here for types)
from google.genai import types as types # Used for structuring messages (Content, Part)

# Google Cloud AI Platform Modules
from google.cloud import storage # Client library for Google Cloud Storage (GCS)


# Other Python Modules
import os # For interacting with the operating system (paths, environment variables)
import warnings # For suppressing warnings
import logging # For controlling logging output
import mimetypes

from dotenv import load_dotenv


# --- Configuration ---
load_dotenv()
project_id = os.environ.get('GOOGLE_CLOUD_PROJECT') # Your GCP Project ID
location = os.environ.get('GOOGLE_CLOUD_LOCATION') # Vertex AI RAG location (can be global for certain setups)
bq_dataset = os.environ.get('BIGQUERY_DATASET')
asset_table_schema = os.environ.get('ASSET_TABLE_SCHEMA')
bq_table = os.environ.get('BIGQUERY_ASSET_TABLE')
#region = os.environ.get('GOOGLE_CLOUD_REGION') # Your GCP region for Vertex AI resources and GCS bucket

# Configuration for Agent Engine
GOOGLE_CLOUD_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
GOOGLE_CLOUD_LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"]
STAGING_BUCKET = os.environ["STAGING_BUCKET"]



# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# --- Environment Setup ---
# Set environment variables required by some Google Cloud libraries
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1" # Instructs the google.genai library to use Vertex AI backend
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = location

# --- Agent Tool Definitions ---
# @title Define Tools for creating a ticket, adding notes to a ticket, add a file to the session, and getting the GCS URI

# --- Tool: add_artifact_from_gcs ---
# This tool is responsible for downloading a file from Google Cloud Storage (GCS)
# and adding it as an artifact to the current agent session. This makes the file's
# content available to other agents and tools in the workflow.
async def add_artifact_from_gcs(uri: str, tool_context: ToolContext) -> str: # Modified signature as per user code
    """
    Adds a specific file from Google Cloud Storage (GCS) to the current session state for agent processing.

    This function takes a GCS URI for a file and wraps it in a `types.Content` object.
    This object is then typically used to make the file's content accessible to the
    agent for tasks like summarization, question answering, or data extraction
    related specifically to that file within the ongoing conversation or session.
    The MIME type is assumed to be inferred by the underlying system or defaults.

    Use this function *after* you have identified a specific GCS URI (e.g., using
    `get_gcs_uri` or similar) that you need the agent to analyze or reference directly.

    Args:
        uri: str - The complete Google Cloud Storage URI of the file to add.
                 Must be in the format "gs://bucket_name/path/to/file.pdf".
                 Example: "gs://my-doc-bucket/reports/q1_report.pdf"

    Returns:
         types.Content - A structured Content object representing the referenced file.
                       This object has `role='user'` and contains a `types.Part`
                       that holds the reference to the provided GCS URI.
                       This Content object can be passed to the agent in subsequent calls.
    """
    
    # Determine the bucket name and blob names
    path_part = uri[len("gs://"):]
    # Split only on the first '/' to separate bucket from the rest
    bucket_name, blob_name = path_part.split('/', 1)

    # Initialize GCS client
    storage_client = storage.Client()

    # Get the bucket and blob objects
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Download the file content as bytes
    file_bytes = blob.download_as_bytes()

    # Determine the mime type of the file based on the file extension
    # Split the URI by the last '/'
    parts = uri.rsplit('/', 1)
    #filename = f'user:{parts[-1]}' # The 'user:' prefix changes the scope of the artifact to the user instead of the session
    filename = parts[-1] # Using the session scope instead of the user scope

    # Detect the MIME type of the file
    mime_type, encoding = mimetypes.guess_type(filename)
    #mime_type = "application/pdf"
    #print(f"Detected MIME type: {mime_type}")

    if mime_type == "application/json":
        mime_type = "text/plain"

    # This part attempts to create a Content object referencing the GCS URI.
    file_artifact = types.Part(
        inline_data=types.Blob(
            data=file_bytes,
            mime_type=mime_type
        )
    )
    
    version = await tool_context.save_artifact(
        filename=filename, artifact=file_artifact
    )

    logging.info(f"Saved artifact {filename} with version {version}")


# --- Tool: list_artifacts ---
# This tool lists all the artifacts that are currently available in the agent session.
# It's useful for debugging and for agents to see what files they have access to.
async def list_artifacts(tool_context: ToolContext) -> str:
    """Tool to list available artifacts for the user."""
    try:
        available_files = await tool_context.list_artifacts()
        if not available_files:
            return "You have no saved artifacts."
        else:
            # Format the list for the user/LLM
            file_list_str = "\n".join([f"- {fname}" for fname in available_files])
            return f"Here are your available Python artifacts:\n{file_list_str}"
    except ValueError as e:
        logging.error(f"Error listing Python artifacts: {e}. Is ArtifactService configured?")
        return "Error: Could not list Python artifacts."
    except Exception as e:
        logging.error(f"An unexpected error occurred during Python artifact list: {e}")
        return "Error: An unexpected error occurred while listing Python artifacts."

    

# Define a tool configuration to block any write operations
tool_config = BigQueryToolConfig(write_mode=WriteMode.ALLOWED)

# Define a credentials config - in this example we are using application default
# credentials
# https://cloud.google.com/docs/authentication/provide-credentials-adc
application_default_credentials, _ = google.auth.default()
credentials_config = BigQueryCredentialsConfig(
    credentials=application_default_credentials
)

# Instantiate a BigQuery toolset
bigquery_toolset = BigQueryToolset(
    credentials_config=credentials_config, bigquery_tool_config=tool_config
)



# --- Agents ---


# --- BQ Agent ---
# This agent is responsible for interacting with BigQuery.
# Its primary function is to take the analysis results from the reasoning_agent
# and insert them into the specified BigQuery table.
# It uses the 'gemini-2.5-pro' model which is optimized for complex reasoning and tool use.
bq_agent = Agent(
    model="gemini-2.5-pro", # Advanced model for complex tasks and reasoning
    name="bq_agent",
    instruction=
    f"""
        You are a marketing expert for your company. You will be provided with a detailed summary of an image or video that you will add to the assets table in BigQuery.
        
        You have access to the following tools and information
        1: bigquery_toolset: Tools to interact with the BigQuery table
        2: asset_table_schema: The schema for the assets table
        3: the BQ dataset and table that you will be using is {project_id}.{bq_dataset}.{bq_table}
        
        
               
        An example workflow would be:
        1: You will be provided with a description of a single video or image
        2: Use the bigquery_toolset tool to identify the schema of the assets table.
            - Note that the asset_id value will be the name of the file
        3: Construct a SQL query to add this asset information to the assets table. 
            - Note that you will only add the asset_id, gcs_uri,and description columns to the table
            - Note that you will write the description exactly as provided to you
        4: Execute the SQL query using the bigquery_toolset tool
        5. Respond with the message: "Asset <name> has been successfully processed and added to the asset table."

        
    """,
    description="Performs reasearch related to a provided question or topic.",
    tools=[
        bigquery_toolset
    ],
)




# --- Reasoning Agent ---
# This agent is the core analysis engine of the workflow.
# It uses a powerful generative model to analyze the content of a digital asset (image or video)
# and generate a detailed, ADA-compliant description.
# It is instructed to be an expert AI assistant specializing in video accessibility.
reasoning_agent = Agent(
    model="gemini-3-pro-preview", # Advanced model for complex tasks and reasoning
    name="reasoning_agent",
    instruction=
    """
        ROLE:

        You are an expert AI assistant specializing in video accessibility for e-commerce. Your role is to act as a "visual interpreter" for users who are blind or have low vision.

        PRIMARY OBJECTIVE:

        Your primary objective is to generate clear and concise audio descriptions for online retailer product videos. The descriptions must provide all key visual information necessary for a user to understand the product's appearance, features, and use, enabling them to make an informed purchasing decision. Your output must comply with the principles of the Americans with Disabilities Act (ADA) and Web Content Accessibility Guidelines (WCAG) for audio descriptions.
        
        You have access to the following tools
        1: load_artifacts: use this to load artifiacts such as files and images
        2: list_artifacts: use this tool to list any artifiacts you have access to
        
               
        Core Description Rules
        1. Prioritize Essential Information:
        Your descriptions must focus on conveying information that a sighted person would use to evaluate the product. Do not describe elements that are purely decorative or irrelevant to the product itself unless they provide essential context.

        2. Describe Key Visuals for Context:
        Focus on actions, settings, and characters that are essential for understanding the video content.

        Actions: Describe what the person in the video is doing with the product (e.g., "A person easily lifts the lightweight suitcase into an overhead bin.").
        Characters/Models: Describe the person interacting with the product in a neutral, objective way (e.g., "A model with long brown hair demonstrates the hairdryer."). Provide characteristics only if relevant to the product's function or fit (e.g., "The jacket fits snugly on the model, who has a broad build.").
        Setting: Briefly describe the environment if it gives context to the product's use (e.g., "The hiking boots are shown on a rocky, muddy trail," or "The blender sits on a modern kitchen counter next to a bowl of fruit.").
        3. Detail Product Appearance and Features:

        Initial Identification: At the first appearance, clearly state what the product is (e.g., "A pair of blue wireless over-ear headphones.").
        Material and Texture: Describe the apparent material and finish (e.g., "The headphones have a matte plastic finish with soft, leather-like earcups."). If a close-up is shown, describe the texture (e.g., "A close-up reveals a woven fabric on the headband.").
        Color and Pattern: Be specific with colors and patterns (e.g., "The ceramic mug is white with a navy-blue geometric pattern."). If multiple colors are shown, list them.
        Size and Scale: Provide a sense of scale by comparing the product to its environment or the person using it (e.g., "The wallet is slightly larger than the palm of the model's hand.").
        Unique Features & Closures: Detail important functional and design elements like logos, zippers, buttons, ports, latches, or unique shapes (e.g., "The backpack features a silver zipper on the front pocket and a discreet, embossed brand logo on the top flap.").
        4. Explain Product Demonstration and Functionality:

        How it Works: Clearly describe any steps shown to operate, assemble, or use the product (e.g., "She presses a single button on the side of the earcup to pause the music.").
        Function in Action: Describe what the product does (e.g., "The vacuum cleaner attachment clicks into place, and its bristles spin as it moves across the carpet.").
        Fit and Movement (for Apparel): For clothing or accessories, describe how the item fits the model and how the fabric moves (e.g., "The maxi dress drapes loosely, flowing behind the model as she walks.").
        5. Announce All On-Screen Text and Graphics:

        Read Verbatim: All text appearing on the screen must be read out exactly as it appears. This includes product names, feature call-outs, specifications, sale information, and contact details.
        Describe Graphics: Describe any meaningful graphics, icons, or charts (e.g., "An icon of a water droplet with a line through it appears, indicating the device is water-resistant.").
        Style and Tone Guidelines
        Be Objective: Describe only what you see. Do not use subjective or interpretive language (e.g., say "The dress is red," not "The dress is a beautiful, festive red.").
        Be Concise: Deliver information efficiently. Integrate descriptions into natural pauses in the video's audio track. Do not talk over dialogue or other essential audio.
        Use Present Tense: Describe the visuals as they happen (e.g., "The model is zipping up the jacket," not "The model zipped up the jacket.").
        Use Simple, Clear Language: Avoid jargon or overly technical terms unless they are part of the product's name or specifications shown on screen.
        Technical and Accessibility Standards
        Synchronization: Your generated description must correspond to the visual events occurring at that moment in the video. 
        Include durations for each description using the following format: [mm:ss - mm:ss] Description. You MUST follow this format in your response.

        The content you generate is part of an automated process to analyze content to comply with ADA and WCAG standards. DO NOT add any tiltes, headings, additional text or formatting to the output.
        An example of a valid response is:
        [00:00 - 00:05] A person is standing in front of a table.
        [00:05 - 00:10] A person is sitting at a desk.
        [00:10 - 00:15] A person is standing in front of a table.

        pass your response to the root_agent to be added to the BigQuery table.
    """,
    description="Performs reasearch related to a provided question or topic.",
    tools=[
        #AgentTool(agent=bq_agent), # Make the bq_agent available as a tool
        list_artifacts,
        load_artifacts,
    ],
)




# --- Root Agent Definition ---
# @title Define the Root Agent with Sub-Agents

# Initialize root agent variables
#root_agent = None
runner_root = None # Initialize runner variable (although runner is created later)

# --- Root Agent ---
# This is the main orchestrator of the agentic workflow.
# It receives the initial trigger from the FastAPI server, coordinates the execution
# of the other agents and tools, and ensures the successful processing of the asset.
# It delegates specific tasks to 'reasoning_agent' and 'bq_agent'.
root_agent = Agent(
    name="root_agent",    # Name for the root agent
    model="gemini-2.5-flash", # Model that supports Audio input and output 
    description="The main coordinator agent. Handles user requests and delegates tasks to specialist sub-agents and tools.", # Description (useful if this agent were itself a sub-agent)
    instruction=                  # The core instructions defining the workflow
    """
        You are the lead Digital Asset Manager ingestion agent. Your goal is to generate an artifact of a digital asset and then delegate to the appropriate agent or tool.

        You have access to specialized tools and sub-agents:
        1. AgentTool `reasoning_agent`: This agent will analyze the digital asset and provide a detailed response.
        2. AgentTool `bq_agent`: This agent will take the output from the `reasoning_agent` and ingest it into the `assets` table in BigQuery.
        
        Your workflow is as follows:
        1. You will receive data representing a single file upload from a Google Cloud Storage event.
        2. Parse this data to extract the "bucket" and "name" of the file.
        3. Construct the full GCS URI in the format "gs://<bucket>/<name>".
        4. Use the `add_artifact_from_gcs` tool with the constructed GCS URI to add the file as an artifact to the session.
        5. Pass the gcs_uri to the `reasoning_agent` AgentTool to analyze the file. 
        6. Pass the reasoning_agent output to the `bq_agent` sub-agent 

    """,
    sub_agents=[
        bq_agent,
        ],
    tools=[
        AgentTool(agent=reasoning_agent), # Make the reasoning_agent available as a tool
        add_artifact_from_gcs,
        list_artifacts,
        load_artifacts,
        load_memory,
    ],
)


# --- Main Execution Block ---
# This block of code is executed when the script is run directly.
# It initializes the Vertex AI SDK and creates a remote agent engine,
# which makes the root_agent and its entire workflow accessible as a deployable service.
# This is how the agent is registered with Vertex AI Agent Engine.
if __name__ == '__main__':
    vertexai.init(
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION,
        staging_bucket=STAGING_BUCKET,
    )
    
    remote_app = agent_engines.create(
    agent_engine=root_agent,
    display_name='video_analysis_agent',
    description='Agent used to inspect and ingest digital assets',
    requirements=[
        "google-cloud-aiplatform[agent_engines,adk]",
        "vertexai",
        "google-cloud-storage",
        "pydantic",
        "python-dotenv",
        "google-genai",   
        ]
    )

    logging.info(f"Created agent engine: {remote_app.resource_name}")
