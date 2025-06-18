from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os
import uuid
import json
import re
import fitz  # PyMuPDF
import docx2txt
import mimetypes
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from google.generativeai import GenerativeModel
import google.generativeai as genai
from PIL import Image
from io import BytesIO
from resume_analyzer import ResumeAnalyzer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini Vision model
vision_model = genai.GenerativeModel("gemini-2.0-flash")

# Initialize FastAPI
app = FastAPI(title="Resume Analyzer API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize resume analyzer
analyzer = ResumeAnalyzer()

# In-memory storage for analysis results
analysis_results = {}

# Resume parsing functions
def parse_docx(file_path):
    return docx2txt.process(file_path)

def parse_pdf_text(file_path):
    """Enhanced PDF text extraction that captures more content"""
    # First try regular text extraction
    text = ""
    with fitz.open(file_path) as doc:
        # Extract text with layout preservation
        for page in doc:
            # Use a higher textpage mode to better preserve layout
            text += page.get_text("text")
        
        # If text seems incomplete (too short), try alternative extraction methods
        if len(text.strip().split()) < 100:  # Too few words
            text = ""
            for page in doc:
                # Try extracting blocks which can preserve more structure
                blocks = page.get_text("blocks")
                blocks.sort(key=lambda b: (b[1], b[0]))  # Sort by y, then x
                for block in blocks:
                    text += block[4] + "\n"
    
    # Clean up the extracted text
    text = text.replace('\u2022', '- ')  # Replace bullet points
    text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
    
    return text.strip()

def extract_text_from_image_or_scanned_pdf(file_path):
    """Improved extraction from images or scanned PDFs"""
    images = []
    if file_path.endswith(".pdf"):
        with fitz.open(file_path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                image = Image.open(BytesIO(img_bytes))
                images.append(image)
    else:
        image = Image.open(file_path)
        images.append(image)

    text = ""
    for idx, img in enumerate(images):
        # Provide more context about what we're looking at
        response = vision_model.generate_content(
            [
                img, 
                "This is a page from a resume. Extract ALL text content from this image, preserving all bullet points, sections, and formatting. Be thorough and don't miss any text, especially bullet points under work experience."
            ]
        )
        extracted = response.text.replace('```', '').strip()
        
        # Add page separator for multi-page documents
        if len(images) > 1:
            text += f"\n--- Page {idx+1} ---\n" + extracted + "\n"
        else:
            text += extracted + "\n"
            
    return text.strip()

def get_text_from_resume(file_path):
    mime, _ = mimetypes.guess_type(file_path)
    if mime:
        if "pdf" in mime:
            text = parse_pdf_text(file_path)
            if len(text.strip()) < 100:  # Very low text — possibly scanned
                text = extract_text_from_image_or_scanned_pdf(file_path)
        elif "word" in mime or file_path.endswith(".docx"):
            text = parse_docx(file_path)
        elif mime.startswith("image/"):
            text = extract_text_from_image_or_scanned_pdf(file_path)
        else:
            raise ValueError("Unsupported file type.")
    else:
        raise ValueError("Unable to determine file type.")
    return text.strip()

def structure_resume_text_to_json(resume_text):
    """Enhanced parsing that ensures all sections are properly identified"""
    prompt = f"""
You are an expert resume parser. Extract ALL information from the following resume text into a comprehensive structured JSON.

Resume Text:
\"\"\"
{resume_text}
\"\"\"

The JSON MUST include ALL of these sections, even if empty:
- personal_info (name, contact, email, phone, location, linkedin, etc.)
- summary (professional summary or objective if present)
- experience (work history with company, role, dates, and ALL bullet points)
- education (degrees, schools, dates, GPA if mentioned)
- skills (technical and soft skills)
- projects (with descriptions and technologies used)
- certifications (if any)
- achievements (awards, recognitions, etc.)
- languages (if any)
- publications (if any)
- volunteer_experience (if any)
- additional_info (anything else noteworthy)

For experience and projects, capture EVERY bullet point mentioned. Do not summarize or omit any points.

For each job in experience section, include:
- company: The company name
- title: Job title
- dates: Employment dates
- location: Work location if mentioned
- bullets: ARRAY of ALL bullet points exactly as they appear in the resume

Return ONLY valid JSON with no explanations. Format properly with clear organization.
"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,  # Lower temperature for more precise extraction
            "max_output_tokens": 8192,  # Ensure enough tokens for complete output
        }
    )
    
    # Clean the response
    cleaned = response.text.strip()
    if cleaned.startswith('```json'):
        cleaned = cleaned[len('```json'):].strip()
    if cleaned.startswith('```'):
        cleaned = cleaned[len('```'):].strip()
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3].strip()
    
    try:
        # Try to parse the JSON
        parsed_json = json.loads(cleaned)
        return parsed_json
    except json.JSONDecodeError:
        # If JSON is invalid, try to fix common formatting issues
        try:
            # Remove line breaks within strings
            cleaned = re.sub(r'(?<=": ").*?(?="(,|\n|}))', lambda m: m.group(0).replace('\n', ' '), cleaned)
            return json.loads(cleaned)
        except:
            # If still failing, try another parsing approach
            fallback_prompt = f"""
Parse the following resume text into a clean JSON format with these sections:
personal_info, summary, experience, education, skills, projects, certifications, achievements.
Keep it simple and ensure valid JSON format. Include ALL bullet points under experience.

Resume:
\"\"\"
{resume_text}
\"\"\"
"""
            fallback_response = model.generate_content(fallback_prompt)
            cleaned_fallback = fallback_response.text.strip()
            if cleaned_fallback.startswith('```json'):
                cleaned_fallback = cleaned_fallback[len('```json'):].strip()
            if cleaned_fallback.startswith('```'):
                cleaned_fallback = cleaned_fallback[len('```'):].strip()
            if cleaned_fallback.endswith('```'):
                cleaned_fallback = cleaned_fallback[:-3].strip()
                
            return json.loads(cleaned_fallback)

def parse_resume(file_path):
    """Enhanced resume parsing with better error handling and text extraction"""
    try:
        raw_text = get_text_from_resume(file_path)
        
        # Check if we have enough text to analyze
        if len(raw_text.strip()) < 100:
            return {
                "error": "Insufficient text extracted from resume. Please try a different file format."
            }, raw_text
            
        # Use enhanced structured JSON parsing
        structured_json = structure_resume_text_to_json(raw_text)
        
        # Verify we have the key sections
        required_sections = ["experience", "education", "skills"]
        missing_sections = [section for section in required_sections if section not in structured_json]
        
        if missing_sections:
            # If key sections are missing, try re-parsing with a simpler approach
            print(f"Missing sections: {missing_sections}. Retrying with simplified parsing...")
            simplified_prompt = f"""
            Parse this resume text into a structured JSON with ONLY these sections:
            experience, education, skills, projects, certifications, achievements.
            For experience, include ALL job entries with their bullet points.
            
            Resume text:
            \"\"\"
            {raw_text}
            \"\"\"
            """
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(simplified_prompt)
            cleaned = response.text.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[len('```json'):].strip()
            if cleaned.startswith('```'):
                cleaned = cleaned[len('```'):].strip()
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3].strip()
                
            try:
                structured_json = json.loads(cleaned)
            except:
                # If still failing, create a minimal structure
                structured_json = {
                    "experience": extract_section_from_text(raw_text, "experience"),
                    "education": extract_section_from_text(raw_text, "education"),
                    "skills": extract_section_from_text(raw_text, "skills")
                }
        
        return structured_json, raw_text
        
    except Exception as e:
        print(f"Error parsing resume: {str(e)}")
        traceback.print_exc()
        return {
            "error": f"Failed to parse resume: {str(e)}"
        }, ""

def extract_section_from_text(text, section_name):
    """Extract a specific section from raw text as fallback"""
    prompt = f"""
    From this resume text, extract ONLY the {section_name.upper()} section.
    Return it as a JSON array of items in this section.
    
    Resume text:
    \"\"\"
    {text}
    \"\"\"
    
    Return ONLY valid JSON, no explanations or markdown.
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    cleaned = response.text.strip()
    
    try:
        # Try to parse as JSON array
        if cleaned.startswith('[') and cleaned.endswith(']'):
            return json.loads(cleaned)
        
        # Try to parse as JSON object (some models might return an object)
        if cleaned.startswith('{') and cleaned.endswith('}'):
            return json.loads(cleaned)
            
        # If it starts with code blocks, clean them
        if cleaned.startswith('```json'):
            cleaned = cleaned[len('```json'):].strip()
        if cleaned.startswith('```'):
            cleaned = cleaned[len('```'):].strip()
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3].strip()
            
        # Try parsing again after cleaning
        if cleaned.startswith('[') and cleaned.endswith(']'):
            return json.loads(cleaned)
            
        # Last resort: return as plain text
        return [cleaned]
    except:
        # Return the text as a single item if parsing fails
        return [f"Could not parse {section_name} section"]

# New endpoint for React frontend compatibility
@app.post("/api/analyze")
async def react_analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(None)
):
    """
    Endpoint compatible with React frontend
    Synchronously processes the resume and returns the analysis results
    """
    try:
        # Check file type
        allowed_extensions = ['.pdf', '.docx', '.doc', '.jpg', '.jpeg', '.png']
        file_ext = os.path.splitext(resume.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid file format. Supported formats: {', '.join(allowed_extensions)}"}
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file_path = temp_file.name
            file_content = await resume.read()
            temp_file.write(file_content)
        
        try:
            # Parse resume
            structured_data, raw_text = parse_resume(temp_file_path)
            
            # Format the parsed data to match the expected structure for the analyzer
            parsed_resume = {
                "raw_content": raw_text,
                "metadata": {
                    "filename": resume.filename,
                    "parser": "gemini"
                },
                "sections": structured_data
            }
            
            # Analyze resume
            analysis_result = analyzer.analyze_full_resume(parsed_resume, job_description)
            
            # Return result directly for the frontend
            return analysis_result
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
    except Exception as e:
        print(f"Error analyzing resume: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to analyze resume: {str(e)}"}
        )

# Also keep the original background processing endpoints
@app.post("/analyze-resume")
async def analyze_resume(
    background_tasks: BackgroundTasks,
    resume_file: UploadFile = File(...),
    job_description: str = Form(None)
):
    """
    Endpoint to upload and analyze a resume
    Returns an analysis ID for fetching results when analysis is complete
    """
    # Check file type
    allowed_extensions = ['.pdf', '.docx', '.doc', '.jpg', '.jpeg', '.png']
    file_ext = os.path.splitext(resume_file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file format. Supported formats: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique ID for this analysis
    analysis_id = str(uuid.uuid4())
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        temp_file_path = temp_file.name
        file_content = await resume_file.read()
        temp_file.write(file_content)
    
    # Store initial status
    analysis_results[analysis_id] = {
        "status": "processing",
        "filename": resume_file.filename,
        "parsed_resume": None,
        "analysis": None
    }
    
    # Process file in background
    background_tasks.add_task(
        process_resume, 
        temp_file_path, 
        analysis_id, 
        resume_file.filename,
        job_description
    )
    
    return {
        "status": "processing",
        "analysis_id": analysis_id,
        "message": "Resume uploaded and being processed. Check status with the analysis_id."
    }

async def process_resume(file_path, analysis_id, filename, job_description):
    """Background task to process resume"""
    try:
        # Parse resume using our parser
        structured_data, raw_text = parse_resume(file_path)
        
        # Format the parsed data to match the expected structure for the analyzer
        parsed_resume = {
            "raw_content": raw_text,
            "metadata": {
                "filename": filename,
                "parser": "gemini"
            },
            "sections": structured_data
        }
        
        if not parsed_resume:
            analysis_results[analysis_id] = {
                "status": "error",
                "filename": filename,
                "error": "Failed to parse resume"
            }
            return
        
        # Analyze resume
        analysis_result = analyzer.analyze_full_resume(parsed_resume, job_description)
        
        # Store result
        analysis_results[analysis_id] = {
            "status": "completed",
            "filename": filename,
            "parsed_resume": parsed_resume,
            "analysis": analysis_result
        }
    except Exception as e:
        analysis_results[analysis_id] = {
            "status": "error",
            "filename": filename,
            "error": str(e)
        }
    finally:
        # Clean up temporary file
        try:
            os.unlink(file_path)
        except:
            pass

@app.get("/analysis/{analysis_id}")
async def get_analysis_result(analysis_id: str):
    """Fetch the result of a resume analysis by ID"""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_results[analysis_id]

@app.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Delete an analysis result"""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    del analysis_results[analysis_id]
    return {"message": "Analysis deleted successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Add demo endpoint for frontend testing
@app.post("/api/analyze/demo")
async def analyze_demo():
    """Endpoint that returns demo data for testing the frontend"""
    return JSONResponse({
        "overall": {
            "score": 78,
            "score_color": "#f1c40f",
            "summary": "Your resume has a good structure but could use improvements in quantifying achievements and tailoring to specific positions.",
            "strengths": [
                "Clear organization of information",
                "Good educational background",
                "Includes relevant technical skills"
            ],
            "weaknesses": [
                "Lacks quantifiable achievements",
                "Generic job descriptions",
                "Professional summary needs strengthening"
            ],
            "impact_score": 65,
            "impact_score_color": "#e67e22",
            "clarity_score": 82,
            "clarity_score_color": "#2ecc71",
        },
        "improvement_plan": {
            "priority_actions": [
                "Add metrics to your experience bullet points",
                "Create a stronger professional summary",
                "Tailor your skills to match job descriptions"
            ],
            "secondary_actions": [
                "Improve formatting consistency",
                "Add links to projects or portfolio",
                "Include certifications if relevant"
            ],
            "section_specific_plan": {
                "experience": [
                    "Start each bullet with a strong action verb",
                    "Add specific metrics and results to your achievements",
                    "Focus on your contributions rather than general responsibilities"
                ],
                "skills": [
                    "Group skills by category (programming languages, tools, etc.)",
                    "Prioritize skills mentioned in job descriptions",
                    "Include proficiency levels for technical skills"
                ],
                "education": [
                    "List relevant coursework if you're a recent graduate",
                    "Include GPA if it's above 3.5",
                    "Add any academic honors or relevant activities"
                ]
            },
            "before_submission_checklist": [
                "Check for spelling and grammar errors",
                "Ensure consistent formatting",
                "Verify contact information is correct",
                "Save as PDF before submitting"
            ],
            "tailoring_strategy": [
                "Identify keywords in job descriptions and incorporate them",
                "Customize your professional summary for each application",
                "Highlight relevant experience based on the job requirements"
            ],
            "visual_improvements": [
                "Use a clean, modern font such as Calibri or Arial",
                "Add subtle color accents to section headers",
                "Ensure consistent spacing between sections"
            ],
            "modern_resume_elements": [
                "Include a skills visualization section",
                "Add a professional headshot if appropriate for your industry",
                "Use a clean, minimalist design with strategic whitespace"
            ],
        },
        "ats_compatibility": {
            "ats_score": 72,
            "ats_score_color": "#f1c40f",
            "parsing_issues": [
                "Complex formatting may cause parsing issues",
                "Non-standard section headers could be misinterpreted"
            ],
            "keyword_analysis": {
                "present_keywords": ["python", "data analysis", "project management", "leadership"],
                "missing_keywords": ["machine learning", "team collaboration", "agile"]
            }
        },
        "sections": {
            "experience": {
                "score": 70,
                "score_color": "#f1c40f",
                "strengths": ["Clear timeline", "Relevant positions"],
                "weaknesses": ["Lacks quantifiable achievements", "Generic descriptions"],
                "improvement_suggestions": [
                    "Add specific metrics to your achievements",
                    "Focus on results rather than responsibilities",
                    "Use stronger action verbs to begin bullet points"
                ]
            },
            "education": {
                "score": 85,
                "score_color": "#2ecc71",
                "strengths": ["Relevant degree", "Good formatting"],
                "weaknesses": ["Missing relevant coursework", "Could add academic achievements"],
                "improvement_suggestions": [
                    "Add relevant coursework if recently graduated",
                    "Include any academic honors or scholarships",
                    "List any significant projects completed during education"
                ]
            },
            "skills": {
                "score": 75,
                "score_color": "#f1c40f",
                "strengths": ["Good range of technical skills", "Includes soft skills"],
                "weaknesses": ["No organization by category", "No indication of proficiency levels"],
                "improvement_suggestions": [
                    "Group skills by category (technical, soft skills, etc.)",
                    "Indicate proficiency levels for technical skills",
                    "Prioritize skills mentioned in target job descriptions"
                ]
            }
        },
        "bullet_point_analysis": [
            {
                "bullet_text": "Managed team of developers on multiple projects",
                "strength": 3,
                "feedback": "This bullet is missing specific metrics and achievements",
                "improved_version": "Led cross-functional team of 8 developers across 3 simultaneous projects, delivering all milestones on time and 10% under budget"
            },
            {
                "bullet_text": "Responsible for customer service and resolving issues",
                "strength": 2,
                "feedback": "Too generic and focuses on responsibilities rather than achievements",
                "improved_version": "Achieved 95% customer satisfaction rating by implementing new issue resolution process, reducing response time by 35%"
            }
        ],
        "analytics": {
            "scores": {
                "overall": {
                    "value": 78,
                    "color": "#f1c40f"
                },
                "ats": {
                    "value": 72,
                    "color": "#f1c40f"
                },
                "sections_avg": {
                    "value": 76.7,
                    "color": "#f1c40f"
                }
            },
            "visualization_data": {
                "section_scores": [
                    {"name": "Experience", "score": 70, "color": "#f1c40f"},
                    {"name": "Education", "score": 85, "color": "#2ecc71"},
                    {"name": "Skills", "score": 75, "color": "#f1c40f"}
                ]
            }
        },
        "ui_components": {
            "theme_colors": {
                "primary": "#3498db",
                "secondary": "#2ecc71",
                "accent": "#e74c3c",
                "neutral": "#34495e",
                "light": "#f5f8fa",
                "dark": "#2c3e50"
            },
            "dashboard_metrics": {
                "total_word_count": 650,
                "bullet_points": 12,
                "section_count": 5,
                "estimated_read_time": 3
            }
        }
    })
@app.post("/analyze")
async def legacy_analyze_endpoint(
    resume: UploadFile = File(...),
    job_description: str = Form(None)
):
    """
    Legacy endpoint redirecting to /api/analyze for backward compatibility
    """
    # Just call the React frontend compatible endpoint
    return await react_analyze_resume(resume, job_description)
@app.get("/")
async def root():
    """Root endpoint with info"""
    return {
        "message": "Resume Analyzer API is running",
        "endpoints": {
            "/api/analyze": "Upload and analyze resume (for React frontend)",
            "/api/analyze/demo": "Get demo analysis data for testing",
            "/analyze-resume": "Upload and analyze resume (background processing)",
            "/analysis/{analysis_id}": "Get analysis results by ID",
            "/health": "Health check"
        },
        "version": "2.0"
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)