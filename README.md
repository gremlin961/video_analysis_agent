# Project Overview: Video Analysis Agent

This document provides a summary of the `video-analysis-agent` project, a FastAPI application designed to process video files from Google Cloud Storage using an agent built with the Google ADK.

## High-Level Architecture

The application is triggered by a file upload to a GCS bucket. The event is processed by a Cloud Task, which then invokes the main application logic. The application uses a multi-agent system to analyze the asset and ingest the metadata into BigQuery.

## Key Components

- **`main.py`**: The main entry point of the application. It's a FastAPI server that handles incoming GCS events and creates Cloud Tasks for asynchronous processing. **Note:** This file is intended as an example of how to use the agentic workflow in a containerized environment and is not meant for production use without further security and error handling improvements.
- **`video_analysis_agent/agent.py`**: This file contains the core logic of the agentic workflow. It defines a `root_agent` that orchestrates the entire process, a `reasoning_agent` for analyzing the digital asset, and a `bq_agent` for ingesting the results into BigQuery.
- **`Dockerfile`**: Defines the container image for the application, which is deployed to Cloud Run.
- **`requirements.txt`**: Lists the Python dependencies for the project.
- **`.env`**: Contains the environment variables for configuring the application and its connection to Google Cloud services.

## Workflow

1.  A file is uploaded to a GCS bucket.
2.  A GCS event is sent to the `/gcs-trigger` endpoint in `main.py`.
3.  A Cloud Task is created to process the event.
4.  The Cloud Task invokes the `/process-asset` endpoint.
5.  The `root_agent` in `agent.py` is executed.
6.  The `root_agent` uses the `add_artifact_from_gcs` tool to make the file available to the agent.
7.  The `reasoning_agent` analyzes the video and generates a summary.
8.  The `bq_agent` ingests the summary and other metadata into a BigQuery table.

## Artifact Management

The agent uses the `GcsArtifactService` to manage temporary files (artifacts) created during the analysis process. These artifacts are stored in a Google Cloud Storage bucket specified by the `ARTIFACTS_BUCKET` environment variable in the `.env` file.

> [!IMPORTANT]
> It is highly recommended to configure a lifecycle policy on the GCS bucket used for artifacts. Since these artifacts are only needed during the processing of a single video, a policy should be set to automatically delete objects after a short period (e.g., 1 day) to prevent unnecessary storage costs.

## Testing and Development

For local testing and development, you can use the `adk web` command provided by the Google ADK. This command starts a local web server with a user interface that allows you to interact with your agent directly, providing a more interactive way to test and debug your agent's behavior. This is a great way to test your agent's logic without having to deploy it to a containerized environment.
