# Resume Analyzer Backend

A FastAPI-based backend service for analyzing resumes using Google's Generative AI.

## Features

- Resume parsing (PDF, DOCX, and image formats)
- AI-powered resume analysis
- Job description matching
- RESTful API endpoints
- Background task processing

## Prerequisites

- Python 3.9+
- Google AI API key
- pip (Python package manager)

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd resume-analyzer-backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your configuration:
```
GOOGLE_API_KEY=your_google_api_key
```

## Running Locally

Start the development server:
```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `POST /analyze-resume`: Upload and analyze a resume
- `GET /analysis/{analysis_id}`: Get analysis results
- `DELETE /analysis/{analysis_id}`: Delete analysis results
- `GET /health`: Health check endpoint

## Deployment

This application is configured for deployment on Render.com. The deployment process is automated through the Procfile and runtime.txt configurations.

## Environment Variables

- `GOOGLE_API_KEY`: Your Google AI API key
- `CORS_ORIGINS`: Allowed frontend origins (comma-separated)

## License

MIT 