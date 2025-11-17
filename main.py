# Copyright 2025 Google, LLC. This software is provided as-is,
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


import os
import uvicorn
import uuid
import json
import base64
import logging
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.adk.runners import Runner
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.sessions import Session, InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from pydantic import BaseModel
from google.genai.types import Content, Part
from google.cloud import tasks_v2
from dotenv import load_dotenv
import video_analysis_agent.agent as video_analysis_agent
import uuid

# Load environment variables from .env file
load_dotenv(dotenv_path="video_analysis_agent/.env")

# --- Configuration ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION_ID = os.environ.get("GOOGLE_CLOUD_LOCATION")
QUEUE_ID = os.environ.get("GOOGLE_CLOUD_TASK_QUEUE_ID")
SERVICE_URL = os.environ.get("GOOGLE_CLOUD_RUN_URL")
SERVICE_EMAIL = os.environ.get("GOOGLE_CLOUD_SERVICE_ACCOUNT_EMAIL")

logging_plugin = LoggingPlugin()

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# --- Cloud Tasks Client Initialization ---
tasks_client = tasks_v2.CloudTasksClient()

# Define the Pydantic model for the GCS event
class GcsEvent(BaseModel):
    bucket: str
    name: str

# --- FastAPI App Initialization ---
# This section initializes the FastAPI application, which will serve as the API layer
# for the agentic workflow. It includes basic metadata for the API.
# This server is intended as an example of how to host the agent in a containerized
# environment and is not recommended for production use without further hardening.
app = FastAPI(
    title="ADK Agent Server",
    description="A FastAPI server for the ADK agent.",
    version="1.0.0",
)

# --- CORS Configuration ---
# Example allowed origins for CORS - WARNING: Using "*" is a security risk in production.
# It's recommended to restrict this to a specific list of trusted domains.
ALLOWED_ORIGINS = ["http://localhost", "http://localhost:8080", "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ADK Runner and Session Initialization ---
# The ADK Runner is the core component that executes the agent logic.
# It requires a session service to manage conversation state and an artifact
# service to handle file-based inputs and outputs.
# For this example, we use in-memory services for simplicity, but for
# production scenarios, you would use persistent storage like a database.
# Use an in-memory session service for simplicity
session_service = InMemorySessionService()
# Use an in-memory artifact service
artifact_service = InMemoryArtifactService()
# Initialize the ADK Runner with the root agent and session service
runner = Runner(
    app_name="video_analysis_agent",
    agent=video_analysis_agent.root_agent,
    plugins=[logging_plugin],
    session_service=session_service,
    artifact_service=artifact_service,
)

# --- API Endpoints ---
@app.get("/")
async def read_root():
    """A simple endpoint to confirm the server is running."""
    return {"message": "Video Analysis Agent is running."}

@app.post("/gcs-trigger")
async def gcs_trigger(event: GcsEvent):
    """
    This endpoint acts as the initial entry point for the workflow, designed to be
    triggered by a Google Cloud Storage (GCS) event (e.g., a new file upload).
    To ensure a quick response to the event trigger and prevent timeouts, this
    function immediately offloads the actual processing to a background task
    by creating a new task in Google Cloud Tasks.

    It acknowledges the GCS event with a 204 No Content response, indicating
    that the event has been received and accepted for processing.
    """
    try:
        # Convert the incoming event object to a JSON string.
        event_str = event.model_dump_json()

        # Construct the full queue path
        parent = tasks_client.queue_path(PROJECT_ID, LOCATION_ID, QUEUE_ID)

        # Construct the task payload for the worker service
        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"{SERVICE_URL}/process-asset",
                "oidc_token": {
                    "service_account_email": f"{SERVICE_EMAIL}",
                    "audience": f"{SERVICE_URL}/process-asset",
                },
                "headers": {"Content-type": "application/json"},
                "body": event_str.encode(),
            },
            "dispatch_deadline": "1800s"
        }

        # Create and queue the task
        tasks_client.create_task(parent=parent, task=task)

        logging.info(f"Successfully created task for event: {event.name}")
        # Immediately return a 204 No Content to acknowledge the event
        return Response(status_code=204)

    except Exception as e:
        logging.error(f"Error creating Cloud Task: {e}")
        # Return a 500 error to indicate failure, causing Pub/Sub to retry.
        raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")


@app.post("/process-asset")
async def process_asset(event: GcsEvent):
    """
    This endpoint is the worker that performs the actual processing of the GCS event.
    It is triggered by a Google Cloud Task created by the /gcs-trigger endpoint.

    The workflow is as follows:
    1. A unique session ID is generated for the transaction.
    2. A new ADK session is created to manage the state of the agent interaction.
    3. The GCS event data is passed to the ADK Runner (`runner.run_async`).
    4. The runner executes the root_agent, which orchestrates the analysis and
       data ingestion process.
    5. The final response from the agent is logged.
    """
    try:
        event_str = event.model_dump_json()
        
        # Generate a unique session ID for this transaction
        session_id = str(uuid.uuid4())
        user_id = "gcs_trigger_user"  # A static user for this automated process

        # Create a new session for this transaction
        await session_service.create_session(
            app_name="video_analysis_agent", session_id=session_id, user_id=user_id
        )

        # Send the GCS event JSON string to the agent to start the ingestion process.
        final_response_text = "Agent did not produce a final response."
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=Content(
                role="user",
                parts=[Part(text=event_str)]
            ),
        ):
            #print(f"Event: {event.content}")
            #print(f"Event: {event.content.parts[0].text}")
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break
        
        logging.info(f"Agent finished processing for session {session_id}. Response: {final_response_text}")
        
        return {"status": "success", "response": final_response_text}

    except Exception as e:
        logging.error(f"Error in agent workflow: {e}")
        # Return a 500 error to signal to Cloud Tasks that the task failed and should be retried.
        raise HTTPException(status_code=500, detail=str(e))

# --- Main Execution ---
# This block of code starts the Uvicorn server when the script is executed directly.
# It listens on all available network interfaces (0.0.0.0) and uses the port
# specified by the PORT environment variable (common in cloud environments like Cloud Run),
# defaulting to 8080 for local development.
if __name__ == "__main__":
    # Use the PORT environment variable provided by Cloud Run, defaulting to 8080
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
