import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import re
from datetime import datetime
import random

# Load environment variables
load_dotenv()

# Configure Google AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCY9tnrZXjEWJMlU39Y6Znd3eMnFQLBbI8")  # Use env var as default
genai.configure(api_key=GOOGLE_API_KEY)

class ResumeAnalyzer:
    def __init__(self, model_name="gemini-2.0-flash"):
        self.model = genai.GenerativeModel(model_name)
        self.version = "2.0.1"
    
    def analyze_full_resume(self, parsed_resume, job_description=None):
        """
        Enhanced analysis of the full resume with comprehensive insights
        
        Args:
            parsed_resume (dict): The parsed resume data containing sections and raw content
            job_description (str, optional): The job description to compare with
            
        Returns:
            dict: Enhanced analysis results
        """
        # Extract sections and raw text
        sections = parsed_resume["sections"]
        resume_text = parsed_resume["raw_content"]
        
        # Track analysis start time
        analysis_start_time = datetime.now()
        
        # Perform various analyses
        analysis_results = {
            "metadata": {
                "version": self.version,
                "analysis_date": datetime.now().isoformat(),
                "has_job_description": bool(job_description),
            },
            "overall": self._analyze_overall_fit(resume_text, job_description),
            "sections": {},
            "ats_compatibility": self.analyze_ats_compatibility(resume_text, job_description),
            "bullet_point_analysis": {},
            "improvement_plan": self._generate_improvement_plan(resume_text, job_description),
        }
        
        # Analyze individual sections with enhanced prompts
        for section_name, section_content in sections.items():
            if section_content:  # Only analyze non-empty sections
                section_analysis = self._analyze_section(section_name, section_content, job_description)
                if section_analysis:
                    analysis_results["sections"][section_name] = section_analysis
                    
                    # Store bullet point feedback separately for experience section
                    if section_name.lower() == "experience" and "bullet_point_feedback" in section_analysis:
                        analysis_results["bullet_point_analysis"] = section_analysis["bullet_point_feedback"]
        
        # Generate additional suggestions
        analysis_results["suggestions"] = self._generate_suggestions(resume_text, job_description)
        
        # Add UI components for frontend
        analysis_results["ui_components"] = self._generate_ui_components(resume_text, job_description)
        
        # Add analytics for dashboard display
        analysis_results["analytics"] = self._generate_analytics_summary(analysis_results)
        
        # Add processing time
        analysis_results["metadata"]["processing_time_ms"] = (datetime.now() - analysis_start_time).total_seconds() * 1000
        
        # Add color codes to scores for UI
        self._add_color_codes_to_scores(analysis_results)
        
        return analysis_results
    
    def _analyze_overall_fit(self, resume_text, job_description):
        """Analyze overall resume fit and quality"""
        prompt = self._create_overall_analysis_prompt(resume_text, job_description)
        response = self.model.generate_content(prompt)
        
        try:
            # Parse the response to structured data
            analysis_text = response.text
            # Clean up the output if it starts/ends with code block markers
            cleaned = analysis_text.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[len('```json'):].strip()
            if cleaned.startswith('```'):
                cleaned = cleaned[len('```'):].strip()
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3].strip()
            
            return json.loads(cleaned)
        except Exception as e:
            print(f"Error parsing overall analysis response: {str(e)}")
            # Return a fallback structure
            return {
                "score": 70,
                "summary": "Unable to generate detailed analysis. Please review the resume manually.",
                "strengths": ["Unable to determine strengths"],
                "weaknesses": ["Unable to determine weaknesses"],
                "recommendations": ["Unable to generate specific recommendations"],
                "impact_score": 65,
                "clarity_score": 70,
                "relevance_score": 60,
            }
    
    def _analyze_section(self, section_name, section_content, job_description):
        """Analyze a specific section of the resume"""
        prompt = self._create_section_analysis_prompt(section_name, section_content, job_description)
        response = self.model.generate_content(prompt)
        
        try:
            # Parse the response to structured data
            analysis_text = response.text
            # Clean up the output if it starts/ends with code block markers
            cleaned = analysis_text.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[len('```json'):].strip()
            if cleaned.startswith('```'):
                cleaned = cleaned[len('```'):].strip()
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3].strip()
            
            return json.loads(cleaned)
        except Exception as e:
            print(f"Error parsing section analysis response for {section_name}: {str(e)}")
            # Return a fallback structure
            return {
                "score": 70,
                "strengths": ["Unable to determine strengths"],
                "weaknesses": ["Unable to determine weaknesses"],
                "improvement_suggestions": ["Unable to generate specific suggestions"],
                "content_quality": 3,
                "formatting_quality": 3
            }
    
    def _generate_suggestions(self, resume_text, job_description):
        """Generate suggestions for resume improvement"""
        prompt = self._create_suggestions_prompt(resume_text, job_description)
        response = self.model.generate_content(prompt)
        
        try:
            # Parse the response to structured data
            suggestions_text = response.text
            # Clean up the output if it starts/ends with code block markers
            cleaned = suggestions_text.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[len('```json'):].strip()
            if cleaned.startswith('```'):
                cleaned = cleaned[len('```'):].strip()
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3].strip()
            
            return json.loads(cleaned)
        except Exception as e:
            print(f"Error parsing suggestions response: {str(e)}")
            # Return a fallback structure
            return {
                "content_suggestions": ["Unable to generate content suggestions"],
                "format_suggestions": ["Unable to generate format suggestions"],
                "general_suggestions": ["Unable to generate general suggestions"]
            }
    
    def _create_overall_analysis_prompt(self, resume_text, job_description):
        """Create an enhanced prompt for overall resume analysis"""
        job_desc_text = f"\nJob Description:\n{job_description}" if job_description else ""
        prompt = f"""
Analyze this resume{job_desc_text} and provide a comprehensive assessment in JSON format.

RESUME:
\"\"\"
{resume_text}
\"\"\"

Evaluate the resume on multiple dimensions including content, clarity, impact, and job fit.

Return a detailed JSON object with these exact fields:
- score: numerical score from 0-100
- summary: concise overall assessment (2-3 sentences)
- strengths: array of 3-5 specific strengths with clear examples from the resume
- weaknesses: array of 3-5 specific areas for improvement with clear rationale
- impact_score: numerical score from 0-100 rating how impactful the content is
- clarity_score: numerical score from 0-100 rating how clear and readable the resume is
- relevance_score: numerical score from 0-100 rating job relevance (if job description provided)
- top_recommended_changes: array of 3-5 highest-priority changes to make, ordered by importance
- section_improvement: object with keys for each major section (summary, experience, education, skills) containing specific improvement suggestions
- keyword_recommendations: array of keywords to add or emphasize based on industry standards or job description
- unique_value_proposition: assessment of what makes this candidate stand out and how to emphasize it better
- visual_appeal_score: numerical score from 0-100 rating the visual organization and professional appearance
- industry_fit: assessment of how well the resume fits industry standards for the candidate's field

The analysis should be specific, actionable, and tailored to this exact resume.

Only return valid JSON - no markdown formatting, comments, or text outside the JSON object.
"""
        return prompt
    
    def _create_section_analysis_prompt(self, section_name, section_content, job_description):
        """Improved prompt for section analysis with bullet point feedback"""
        job_desc_text = f"\nJob Description:\n{job_description}" if job_description else ""
        
        # Format section content appropriately
        if isinstance(section_content, list):
            formatted_content = "\n".join([str(item) for item in section_content])
        elif isinstance(section_content, dict):
            # For experience sections that might contain job entries as dictionaries
            formatted_content = json.dumps(section_content, indent=2)
        else:
            formatted_content = str(section_content)
        
        # Special handling for experience section to get bullet-by-bullet feedback
        if section_name.lower() == "experience":
            prompt = f"""
Analyze the {section_name} section of this resume{job_desc_text} with special focus on each bullet point.

{section_name.upper()} SECTION:
\"\"\"
{formatted_content}
\"\"\"

Return a JSON object with these fields:
- score: numerical score from 0-100
- strengths: array of 2-4 strengths of this section
- weaknesses: array of 2-4 areas for improvement in this section
- bullet_point_feedback: array of objects analyzing each bullet point with these properties:
  - bullet_text: the original bullet point text
  - strength: is this bullet strong or weak (1-5 scale, where 5 is strongest)
  - feedback: specific suggestion to improve this bullet point
  - improved_version: rewritten version of this bullet point that is stronger, more impactful, and quantified
  - impact_words: array of powerful words or phrases that could strengthen this bullet
  - metrics_added: boolean indicating if metrics were added to the improved version

- improvement_suggestions: array of 2-4 specific suggestions for the entire section
- industry_alignment: assessment of how well this section aligns with industry expectations
- experience_showcase: assessment of how effectively the experience demonstrates skills relevant to target roles

For bullet point feedback:
1. Focus on making each bullet RESULTS-ORIENTED and QUANTIFIED
2. Add specific metrics and achievements where possible
3. Use strong action verbs and remove filler words
4. Ensure proper formatting and consistent tense

Only return valid JSON - no markdown formatting or text outside the JSON.
"""
        elif section_name.lower() in ["skills", "technical skills", "competencies"]:
            prompt = f"""
Analyze the {section_name} section of this resume{job_desc_text} with special focus on technical skill organization and relevance.

{section_name.upper()} SECTION:
\"\"\"
{formatted_content}
\"\"\"

Return a JSON object with these fields:
- score: numerical score from 0-100
- strengths: array of 2-4 specific strengths of this section
- weaknesses: array of 2-4 specific areas for improvement in this section
- improvement_suggestions: array of 2-4 specific suggestions with examples
- content_quality: assessment of how well the content represents the candidate's qualifications (1-5 scale)
- formatting_quality: assessment of how well the section is formatted and structured (1-5 scale)
- skill_categorization: suggested way to categorize these skills (by proficiency, by type, etc.)
- missing_skills: array of potentially important skills missing based on industry standards or job description
- skill_relevance: assessment of how relevant these skills are to the target position or industry

For the suggestions:
1. Be specific and actionable
2. Focus on organization, categorization, and highlighting the most impressive/relevant skills
3. Suggest skills to add or remove based on relevance

Only return valid JSON - no markdown formatting or text outside the JSON.
"""
        else:
            prompt = f"""
Analyze the {section_name} section of this resume{job_desc_text} and provide detailed assessment in JSON format.

{section_name.upper()} SECTION:
\"\"\"
{formatted_content}
\"\"\"

Return a JSON object with these fields:
- score: numerical score from 0-100
- strengths: array of 2-4 specific strengths of this section
- weaknesses: array of 2-4 specific areas for improvement in this section
- improvement_suggestions: array of 2-4 specific suggestions with examples
- content_quality: assessment of how well the content represents the candidate's qualifications (1-5 scale)
- formatting_quality: assessment of how well the section is formatted and structured (1-5 scale)
- section_relevance: how important this section is for the overall resume effectiveness
- optimization_tips: specific ways to make this section more impactful and relevant

For the suggestions:
1. Be specific and actionable
2. Provide concrete examples where possible
3. Focus on both content and presentation

Only return valid JSON - no markdown formatting or text outside the JSON.
"""
        
        return prompt
    
    def _create_suggestions_prompt(self, resume_text, job_description):
        """Create prompt for generating improvement suggestions"""
        job_desc_text = f"\nJob Description:\n{job_description}" if job_description else ""
        prompt = f"""
Generate specific, actionable suggestions to improve this resume{job_desc_text}. Return in JSON format.

Resume:
\"\"\"
{resume_text}
\"\"\"

Return only a JSON object with these exact fields:
- content_suggestions: array of 3-5 suggestions to improve content
- format_suggestions: array of 2-3 suggestions to improve formatting
- general_suggestions: array of 2-3 general improvement tips
- professional_branding_tips: array of 2-3 suggestions to improve personal branding
- digital_presence_suggestions: array of 2-3 tips for improving online professional presence
- modern_resume_techniques: array of 2-3 current best practices for modern resumes

Each suggestion should be specific, actionable, and tailored to this resume.

Only return valid JSON - no markdown formatting, comments, explanations or text outside the JSON object.
"""
        return prompt

    def _generate_improvement_plan(self, resume_text, job_description):
        """Generate a structured improvement plan for the resume"""
        job_desc_text = f"\nJob Description:\n{job_description}" if job_description else ""
        
        # First, extract section names from the resume to ensure section-specific plans include all sections
        section_names = self._extract_section_names(resume_text)
        
        # Extract section content to provide better context for improvement suggestions
        sections_content = self._extract_section_content(resume_text)
        
        # Prepare section content for prompt
        sections_context = "\n\n".join([f"{name.upper()}:\n{content}" for name, content in sections_content.items()])
        
        prompt = f"""
Create a detailed, tailored improvement plan for this resume.{job_desc_text}

RESUME SECTIONS:
{sections_context}

Provide specific, actionable recommendations for each resume section. 
Return a JSON object with these fields:
- priority_actions: array of 3-5 highest priority changes to make immediately (be specific)
- secondary_actions: array of 3-5 additional improvements to make (be specific)
- section_specific_plan: object with TAILORED improvement steps for each resume section
- tailoring_strategy: array of specific steps to better tailor this resume for the target job
- before_submission_checklist: array of items to check before submitting
- visual_improvements: array of 3-5 suggestions to improve the visual appeal of the resume
- modern_resume_elements: array of 2-4 elements to make the resume more contemporary
- digital_integration: array of 2-3 ways to integrate digital elements (QR codes, links, etc.)

For the section_specific_plan, create TAILORED suggestions for EACH section based on its actual content:
1. DO NOT use generic suggestions like "Review this section for completeness"
2. DO provide specific, actionable recommendations like "Add metrics to your project achievements" 
3. Suggestions should be DIFFERENT for EACH section
4. Name the sections using their EXACT names from the resume
5. Format as an object with section names as keys and arrays of 2-4 suggestions as values
6. For technical skills sections, provide specific organization strategies
7. For experience sections, focus on impact and quantifiable achievements
8. For project sections, emphasize technologies and methodologies

Example section_specific_plan format:
{{
  "skills": [
    "Group technical skills by proficiency level (Expert, Advanced, Familiar)",
    "Add specific versions of technologies (e.g., 'Python 3.8' instead of just 'Python')",
    "Include relevant cloud platforms that match the job description"
  ],
  "experience": [
    "Start each bullet with powerful action verbs like 'Implemented' or 'Architected'",
    "Quantify achievements with metrics (e.g., 'Reduced processing time by 40%')",
    "Focus more on outcomes and less on responsibilities"
  ]
}}

Only return valid JSON without any explanation or markdown formatting.
"""

        response = self.model.generate_content(prompt)
        cleaned = response.text.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[len('```json'):].strip()
        if cleaned.startswith('```'):
            cleaned = cleaned[len('```'):].strip()
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3].strip()
        
        try:
            plan = json.loads(cleaned)
            # Validate and ensure the section_specific_plan has proper structure with arrays
            validated_plan = self._validate_improvement_plan(plan, section_names)
            
            # Add UI elements
            self._add_ui_elements_to_plan(validated_plan)
            
            return validated_plan
        except Exception as e:
            print(f"Error parsing improvement plan: {str(e)}")
            default_plan = self._create_default_improvement_plan(sections_content.keys())
            self._add_ui_elements_to_plan(default_plan)
            return default_plan
    
    def _add_ui_elements_to_plan(self, plan):
        """Add UI elements to the improvement plan"""
        # Add IDs and priorities to priority actions
        if "priority_actions" in plan and isinstance(plan["priority_actions"], list):
            for i, action in enumerate(plan["priority_actions"]):
                if isinstance(action, str):
                    plan["priority_actions"][i] = {
                        "text": action,
                        "id": f"priority-{i+1}",
                        "priority": "high",
                        "priority_color": self._get_priority_color("high")
                    }
        
        # Add IDs and priorities to secondary actions
        if "secondary_actions" in plan and isinstance(plan["secondary_actions"], list):
            for i, action in enumerate(plan["secondary_actions"]):
                if isinstance(action, str):
                    plan["secondary_actions"][i] = {
                        "text": action,
                        "id": f"secondary-{i+1}",
                        "priority": "medium",
                        "priority_color": self._get_priority_color("medium")
                    }
        
        # Add IDs to section specific plans
        if "section_specific_plan" in plan and isinstance(plan["section_specific_plan"], dict):
            for section, suggestions in plan["section_specific_plan"].items():
                if isinstance(suggestions, list):
                    for i, suggestion in enumerate(suggestions):
                        if isinstance(suggestion, str):
                            suggestions[i] = {
                                "text": suggestion,
                                "id": f"{section}-{i+1}"
                            }
        
        # Add IDs to checklist items
        if "before_submission_checklist" in plan and isinstance(plan["before_submission_checklist"], list):
            for i, item in enumerate(plan["before_submission_checklist"]):
                if isinstance(item, str):
                    plan["before_submission_checklist"][i] = {
                        "text": item,
                        "id": f"checklist-{i+1}",
                        "checked": False
                    }
                    
        # Add UI elements to other lists
        additional_lists = ["visual_improvements", "modern_resume_elements", "digital_integration", "tailoring_strategy"]
        for list_name in additional_lists:
            if list_name in plan and isinstance(plan[list_name], list):
                for i, item in enumerate(plan[list_name]):
                    if isinstance(item, str):
                        plan[list_name][i] = {
                            "text": item,
                            "id": f"{list_name.replace('_', '-')}-{i+1}"
                        }

    def _extract_section_content(self, resume_text):
        """Extract section content from the resume text"""
        # Dictionary to store section names and their content
        sections = {}
        
        # Common resume section headers to look for
        common_headers = [
            "summary", "objective", "profile", "about me", "experience", "work experience", 
            "employment history", "education", "skills", "technical skills", "projects", 
            "certifications", "achievements", "publications", "languages", "interests",
            "volunteer experience", "professional experience", "leadership", "coursework",
            "contact", "references", "activities", "machine learning", "web development"
        ]
        
        # Convert resume text to lowercase for case-insensitive matching
        resume_lower = resume_text.lower()
        lines = resume_text.split('\n')
        
        current_section = None
        section_content = []
        
        # First pass: identify potential section headers
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check if this line could be a section header
            is_header = False
            matched_header = None
            
            # Check if line matches common headers
            for header in common_headers:
                if line_lower == header or line_lower == header + ":" or line_lower.startswith(header + " "):
                    is_header = True
                    matched_header = header
                    break
            
            # If no exact match but line is capitalized and short, it might be a header
            if not is_header and len(line.strip()) < 30 and line.strip() and line.strip()[0].isupper():
                # Check if followed by empty line or content
                if i < len(lines) - 1 and lines[i + 1].strip() and not lines[i + 1].strip()[0].isupper():
                    is_header = True
                    matched_header = line_lower
            
            if is_header:
                # Save previous section if exists
                if current_section and section_content:
                    sections[current_section] = "\n".join(section_content)
                
                # Start new section
                current_section = matched_header if matched_header else line_lower
                current_section = current_section.strip(':').strip()
                section_content = []
            elif current_section is not None:
                # Add line to current section
                section_content.append(line)
        
        # Save the last section
        if current_section and section_content:
            sections[current_section] = "\n".join(section_content)
        
        # Format section names for consistency
        formatted_sections = {}
        for name, content in sections.items():
            # Clean the section name and format it
            clean_name = name.lower().strip().replace(" ", "_")
            formatted_sections[clean_name] = content.strip()
        
        return formatted_sections

    def _create_default_improvement_plan(self, section_names):
        """Create a default improvement plan with specific suggestions for each section type"""
        
        section_specific_plan = {}
        
        # More specific default suggestions based on section type
        default_suggestions = {
            "contact_information": [
                "Add your LinkedIn profile URL",
                "Use a professional email address (@gmail.com rather than @yahoo.com or @hotmail.com)",
                "Remove personal details like age, marital status, or ID numbers"
            ],
            "summary": [
                "Add a brief summary or objective statement highlighting your key skills and career goals",
                "Tailor this section to the specific role you're applying for",
                "Include your years of experience and top technical skills"
            ],
            "profile": [
                "Keep your profile concise (3-5 lines maximum)",
                "Focus on your unique value proposition and career achievements",
                "Mention your career goals that align with the position"
            ],
            "experience": [
                "Start each bullet with strong action verbs (e.g., Implemented, Developed, Led)",
                "Quantify your achievements with metrics and percentages",
                "Focus more on results and less on responsibilities"
            ],
            "work_experience": [
                "Include company descriptions for less-known organizations",
                "Add measurable achievements for each role",
                "Ensure chronological order with most recent experience first"
            ],
            "education": [
                "Include your GPA if it's above 3.5",
                "Add relevant coursework that aligns with the job requirements",
                "List academic achievements, honors, or scholarships"
            ],
            "skills": [
                "Group skills by category (programming languages, frameworks, tools)",
                "Indicate proficiency level for technical skills",
                "Ensure skills mentioned in the job description appear prominently"
            ],
            "technical_skills": [
                "Organize technical skills by proficiency level",
                "Include versions of technologies where relevant (e.g., Python 3.x, React 17)",
                "Add relevant cloud platforms and tools"
            ],
            "projects": [
                "Include links to GitHub repositories or live demos",
                "Highlight your specific contributions to each project",
                "Describe the technologies and methodologies used"
            ],
            "certifications": [
                "Include certification dates and expiration if applicable",
                "Add certification numbers for verification purposes",
                "List certifications in order of relevance to the target role"
            ],
            "languages": [
                "Indicate proficiency level for each language",
                "Use standard descriptors (Native, Fluent, Professional, Basic)",
                "Only include languages with at least basic proficiency"
            ],
            "linkedin_github": [
                "Add your complete LinkedIn URL with customized handle",
                "Include GitHub profile showing your contributions and projects",
                "Consider adding other relevant professional profiles"
            ],
            "web_development": [
                "List specific frontend and backend technologies you're proficient in",
                "Mention frameworks and libraries you've worked with",
                "Include experience with responsive design and performance optimization"
            ],
            "machine_learning": [
                "Specify ML libraries and frameworks you've used (TensorFlow, PyTorch, etc.)",
                "Mention specific ML techniques and algorithms you're familiar with",
                "Include experience with data preprocessing and model deployment"
            ],
            "volunteer_experience": [
                "Highlight leadership roles or responsibilities in volunteer work",
                "Connect volunteer experience to relevant professional skills",
                "Include quantifiable achievements from volunteer activities"
            ]
        }
        
        # For each section name, add specific suggestions
        for section in section_names:
            clean_section = str(section).lower().strip().replace(" ", "_")
            
            # Try to match with our default suggestions
            matched = False
            for default_key in default_suggestions:
                if default_key in clean_section or clean_section in default_key:
                    section_specific_plan[section] = default_suggestions[default_key]
                    matched = True
                    break
            
            # If no match found, use generic but somewhat tailored suggestions
            if not matched:
                section_specific_plan[section] = [
                    f"Make your {section.replace('_', ' ')} more specific and achievement-oriented",
                    f"Ensure {section.replace('_', ' ')} information aligns with job requirements",
                    f"Use industry-standard terminology in your {section.replace('_', ' ')} section"
                ]
        
        return {
            "priority_actions": [
                "Quantify achievements in your experience section with specific metrics",
                "Tailor your skills section to match the job description keywords",
                "Create a stronger professional summary highlighting your unique value"
            ],
            "secondary_actions": [
                "Improve formatting consistency throughout the document",
                "Add LinkedIn profile and GitHub links to your contact information",
                "Include specific versions of technical skills and technologies"
            ],
            "section_specific_plan": section_specific_plan,
            "tailoring_strategy": [
                "Analyze the job description for key requirements and reflect them in your resume",
                "Prioritize skills and experiences most relevant to the target position",
                "Use industry terminology and keywords from the job posting"
            ],
            "before_submission_checklist": [
                "Check for spelling and grammar errors throughout the document",
                "Ensure consistent formatting (font, bullet points, spacing)",
                "Verify that all links are working and contact information is correct",
                "Save as PDF to preserve formatting across different systems"
            ],
            "visual_improvements": [
                "Use a clean, modern font such as Calibri, Arial, or Helvetica",
                "Add subtle color accents to section headers for visual interest",
                "Ensure consistent spacing between sections and adequate margins",
                "Use bullet points for better readability and visual organization"
            ],
            "modern_resume_elements": [
                "Add a professional headshot if appropriate for your industry",
                "Include a QR code linking to your portfolio or LinkedIn profile",
                "Create a skills visualization using simple graphs or rating systems",
                "Use a clean, minimalist design with strategic whitespace"
            ],
            "digital_integration": [
                "Add clickable links to your portfolio, GitHub, and LinkedIn profiles",
                "Include QR codes for quick access to your online presence",
                "Optimize the document with ATS-friendly keywords while maintaining readability"
            ]
        }

    def _validate_improvement_plan(self, plan, section_names):
        """Validate and fix the improvement plan structure"""
        
        if not isinstance(plan, dict):
            return self._create_default_improvement_plan(section_names)
            
        # Ensure required keys exist
        required_keys = ["priority_actions", "secondary_actions", "section_specific_plan", 
                         "tailoring_strategy", "before_submission_checklist"]
        
        for key in required_keys:
            if key not in plan:
                if key in ["priority_actions", "secondary_actions", "before_submission_checklist"]:
                    plan[key] = []
                elif key == "tailoring_strategy":
                    plan[key] = ["Focus on matching keywords from job descriptions", 
                                "Highlight relevant experience for the specific role"]
                elif key == "section_specific_plan":
                    plan[key] = {}
                
        # Ensure section_specific_plan exists and has entries for all sections
        if not isinstance(plan["section_specific_plan"], dict):
            plan["section_specific_plan"] = {}
        
        # Check if section names are properly formatted
        formatted_sections = {}
        for section in plan["section_specific_plan"]:
            formatted_name = section.lower().strip().replace(" ", "_")
            formatted_sections[formatted_name] = plan["section_specific_plan"][section]
        
        plan["section_specific_plan"] = formatted_sections
        
        # Ensure each section contains an array of suggestions (not a string)
        for section, suggestions in plan["section_specific_plan"].items():
            if not isinstance(suggestions, list):
                value = suggestions
                if isinstance(value, str):
                    plan["section_specific_plan"][section] = [value]
                else:
                    plan["section_specific_plan"][section] = [
                        f"Review {section.replace('_', ' ')} for completeness and accuracy",
                        f"Ensure {section.replace('_', ' ')} is relevant to your target role"
                    ]
        
        # Ensure tailoring_strategy is a list
        if isinstance(plan["tailoring_strategy"], str):
            plan["tailoring_strategy"] = [plan["tailoring_strategy"]]
            
        # Add missing modern elements if not present
        additional_keys = ["visual_improvements", "modern_resume_elements", "digital_integration"]
        for key in additional_keys:
            if key not in plan:
                if key == "visual_improvements":
                    plan[key] = [
                        "Use a clean, modern font such as Calibri, Arial, or Helvetica",
                        "Add subtle color accents to section headers for visual interest",
                        "Ensure consistent spacing between sections and adequate margins"
                    ]
                elif key == "modern_resume_elements":
                    plan[key] = [
                        "Create a skills visualization using simple graphs or rating systems",
                        "Use a clean, minimalist design with strategic whitespace",
                        "Include a brief professional profile/summary at the top"
                    ]
                elif key == "digital_integration":
                    plan[key] = [
                        "Add clickable links to your portfolio, GitHub, and LinkedIn profiles",
                        "Include QR codes for quick access to your online presence"
                    ]
        
        return plan

    def _extract_section_names(self, resume_text):
        """Extract likely section names from the resume text"""
        # Try to extract actual sections from the resume
        sections_content = self._extract_section_content(resume_text)
        if sections_content:
            return list(sections_content.keys())
        
        # Fallback to common sections if extraction fails
        common_sections = [
            "contact_information", "summary", "experience", "education", 
            "skills", "technical_skills", "projects", "certifications",
            "achievements", "publications", "languages", "interests",
            "volunteer_experience", "leadership", "relevant_coursework"
        ]
        
        return common_sections
            
    def analyze_ats_compatibility(self, resume_text, job_description=None):
        """Analyze how well the resume will perform with ATS systems"""
        job_desc_text = f"\nJob Description:\n{job_description}" if job_description else ""
        
        prompt = f"""
Analyze this resume for Applicant Tracking System (ATS) compatibility.{job_desc_text}

RESUME:
\"\"\"
{resume_text}
\"\"\"

Evaluate how well the resume will perform in automated ATS screening systems. Consider:
1. Proper heading formats and section naming
2. Use of standard section titles
3. Keyword optimization
4. Formatting issues that might cause parsing problems
5. File format compatibility
6. Overall keyword density and relevance

Return a JSON object with these fields:
- ats_score: numerical score from 0-100 representing ATS compatibility
- parsing_issues: array of specific formatting or structure issues that might cause ATS problems
- keyword_analysis: assessment of keyword usage and optimization
  - present_keywords: array of important keywords found in the resume
  - missing_keywords: array of recommended keywords to add (especially from job description if provided)
- format_compatibility: assessment of the resume format for ATS systems
- optimization_suggestions: array of specific suggestions to improve ATS performance
- keyword_density: assessment of keyword frequency and distribution
- section_headers_analysis: analysis of how well section headers follow ATS-friendly conventions
- file_format_recommendation: recommended file format for ATS submissions

Only return valid JSON without any explanation or text outside the JSON object.
"""

        response = self.model.generate_content(prompt)
        cleaned = response.text.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[len('```json'):].strip()
        if cleaned.startswith('```'):
            cleaned = cleaned[len('```'):].strip()
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3].strip()
        
        try:
            result = json.loads(cleaned)
            
            # Add UI elements for ATS parsing issues
            if "parsing_issues" in result and isinstance(result["parsing_issues"], list):
                for i, issue in enumerate(result["parsing_issues"]):
                    if isinstance(issue, str):
                        result["parsing_issues"][i] = {
                            "text": issue,
                            "id": f"issue-{i+1}",
                            "severity": "high" if i < 2 else "medium",
                            "severity_color": self._get_priority_color("high" if i < 2 else "medium")
                        }
                        
            # Add UI elements for optimization suggestions
            if "optimization_suggestions" in result and isinstance(result["optimization_suggestions"], list):
                for i, suggestion in enumerate(result["optimization_suggestions"]):
                    if isinstance(suggestion, str):
                        result["optimization_suggestions"][i] = {
                            "text": suggestion,
                            "id": f"ats-opt-{i+1}",
                            "priority": "high" if i < 2 else "medium",
                            "priority_color": self._get_priority_color("high" if i < 2 else "medium")
                        }
            
            return result
            
        except Exception as e:
            # Return a fallback structure
            return {
                "ats_score": 70,
                "parsing_issues": ["Unable to determine specific parsing issues"],
                "keyword_analysis": {
                    "present_keywords": ["Unable to analyze keywords"],
                    "missing_keywords": ["Unable to determine missing keywords"]
                },
                "format_compatibility": "Unable to assess format compatibility",
                "optimization_suggestions": ["Unable to generate specific optimization suggestions"]
            }

    def enhance_bullet_points(self, bullet_points, job_description=None):
        """
        Generate improved versions of resume bullet points
        
        Args:
            bullet_points (list): List of existing bullet point strings
            job_description (str, optional): Job description for tailoring
            
        Returns:
            list: Dictionary of original bullets mapped to enhanced versions
        """
        job_desc_text = f"\nJob Description:\n{job_description}" if job_description else ""
        
        # Format the bullet points into a string
        bullet_text = "\n".join([f"- {bp}" for bp in bullet_points])
        
        prompt = f"""
Enhance these resume bullet points to make them more impactful, quantified, and results-oriented.{job_desc_text}

ORIGINAL BULLET POINTS:
{bullet_text}

For each bullet point:
1. Add SPECIFIC metrics, numbers or percentages wherever possible
2. Use strong ACTION VERBS at the beginning
3. Focus on ACCOMPLISHMENTS and RESULTS, not just responsibilities
4. Remove filler words and unnecessary details
5. Ensure proper formatting and consistent tense (preferably past tense)
6. Include relevant SKILLS or TECHNOLOGIES mentioned in the job description
7. Add specific impact of your work (e.g., effect on business metrics, user satisfaction, etc.)
8. Include the skills or tools used to accomplish the task

Return a JSON object with this format:
{{
  "enhanced_bullets": [
    {{
      "original": "original bullet text",
      "enhanced": "improved bullet text",
      "explanation": "brief explanation of changes made and why they're effective",
      "impact_score": numeric score from 1-5 rating how impactful the enhanced version is,
      "action_verb": "the strong action verb used at the beginning",
      "metrics_added": boolean indicating if quantifiable metrics were added
    }},
    // More entries...
  ]
}}

Only return valid JSON - no markdown formatting or text outside the JSON.
"""

        response = self.model.generate_content(prompt)
        cleaned = response.text.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[len('```json'):].strip()
        if cleaned.startswith('```'):
            cleaned = cleaned[len('```'):].strip()
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3].strip()
        
        try:
            result = json.loads(cleaned)
            
            # Add UI elements
            if "enhanced_bullets" in result and isinstance(result["enhanced_bullets"], list):
                for i, bullet in enumerate(result["enhanced_bullets"]):
                    if "impact_score" in bullet:
                        bullet["impact_score_color"] = self._get_score_color(bullet["impact_score"] * 20)
                    bullet["id"] = f"bullet-{i+1}"
                    
            return result["enhanced_bullets"]
            
        except Exception as e:
            # Fall back to a simpler approach if JSON parsing fails
            return [{"original": bp, "enhanced": bp, "explanation": "Unable to process enhancement", 
                    "impact_score": 3, "impact_score_color": self._get_score_color(60),
                    "action_verb": "", "metrics_added": False, "id": f"bullet-{i+1}"} 
                    for i, bp in enumerate(bullet_points)]
    
    def _generate_ui_components(self, resume_text, job_description):
        """Generate UI-specific components for a modern interface"""
        # Extract key metrics for dashboard visualization
        sections = self._extract_section_content(resume_text)
        
        # Track resume content metrics
        word_count = len(resume_text.split())
        bullet_points = len(re.findall(r'•|\*|-|\d+\.', resume_text))
        section_count = len(sections)
        
        # Create chart data for visualization
        chart_data = {
            "resume_composition": [
                {"section": "Experience", "percentage": 35},
                {"section": "Skills", "percentage": 20},
                {"section": "Education", "percentage": 15},
                {"section": "Projects", "percentage": 15},
                {"section": "Other", "percentage": 15}
            ],
            "keyword_density": []
        }
        
        # Create comparison data if job description is available
        if job_description:
            common_words = set([w.lower() for w in resume_text.split()]) & set([w.lower() for w in job_description.split()])
            job_match_score = min(len(common_words) * 3, 100)
            chart_data["job_match"] = {
                "score": job_match_score,
                "color": self._get_score_color(job_match_score)
            }
            
            # Extract potential keywords from job description
            common_skills = ["python", "javascript", "react", "node.js", "aws", "docker", "sql", "java", "c++", "linux"]
            job_skills = [skill for skill in common_skills if skill.lower() in job_description.lower()]
            resume_skills = [skill for skill in job_skills if skill.lower() in resume_text.lower()]
            
            chart_data["skill_match"] = {
                "matched": len(resume_skills),
                "total": len(job_skills),
                "percentage": int(len(resume_skills) / max(len(job_skills), 1) * 100),
                "skills": {
                    "matched": resume_skills,
                    "missing": [s for s in job_skills if s not in resume_skills]
                }
            }
        
        # Generate theme colors for UI customization
        theme_colors = {
            "primary": "#3498db",
            "secondary": "#2ecc71",
            "accent": "#e74c3c",
            "neutral": "#34495e",
            "light": "#ecf0f1",
            "dark": "#2c3e50"
        }
        
        # Dashboard metrics
        dashboard_metrics = {
            "total_word_count": word_count,
            "bullet_points": bullet_points,
            "section_count": section_count,
            "estimated_read_time": max(1, int(word_count / 200))  # in minutes
        }
        
        # Compile UI components
        ui_components = {
            "dashboard_metrics": dashboard_metrics,
            "chart_data": chart_data,
            "theme_colors": theme_colors,
            "layout_recommendation": "two_column",  # options: one_column, two_column, hybrid
            "visualizations": [
                "skill_spider_chart",
                "experience_timeline",
                "keyword_cloud",
                "comparison_radar_chart"
            ]
        }
        
        return ui_components
    
    def _generate_analytics_summary(self, analysis_results):
        """Generate summary analytics from analysis results for dashboard display"""
        analytics = {
            "scores": {},
            "improvement_areas": [],
            "strengths": [],
            "visualization_data": {}
        }
        
        # Extract scores
        if "overall" in analysis_results and "score" in analysis_results["overall"]:
            analytics["scores"]["overall"] = {
                "value": analysis_results["overall"]["score"],
                "color": self._get_score_color(analysis_results["overall"]["score"])
            }
            
        if "ats_compatibility" in analysis_results and "ats_score" in analysis_results["ats_compatibility"]:
            analytics["scores"]["ats"] = {
                "value": analysis_results["ats_compatibility"]["ats_score"],
                "color": self._get_score_color(analysis_results["ats_compatibility"]["ats_score"])
            }
            
        # Calculate average section score
        section_scores = []
        for section_name, section_data in analysis_results.get("sections", {}).items():
            if "score" in section_data and isinstance(section_data["score"], (int, float)):
                section_scores.append(section_data["score"])
                
        if section_scores:
            avg_section_score = sum(section_scores) / len(section_scores)
            analytics["scores"]["sections_avg"] = {
                "value": round(avg_section_score, 1),
                "color": self._get_score_color(avg_section_score)
            }
            
        # Extract improvement areas
        if "overall" in analysis_results and "weaknesses" in analysis_results["overall"]:
            for weakness in analysis_results["overall"]["weaknesses"][:3]:
                analytics["improvement_areas"].append({
                    "text": weakness,
                    "priority": "high"
                })
                
        # Extract strengths
        if "overall" in analysis_results and "strengths" in analysis_results["overall"]:
            for strength in analysis_results["overall"]["strengths"][:3]:
                analytics["strengths"].append({
                    "text": strength,
                    "impact": "high"
                })
                
        # Generate data for visualizations
        visualization_data = {
            "section_scores": [],
            "skill_analysis": [],
            "improvement_priority": []
        }
        
        # Section scores for radar chart
        for section_name, section_data in analysis_results.get("sections", {}).items():
            if "score" in section_data and isinstance(section_data["score"], (int, float)):
                visualization_data["section_scores"].append({
                    "name": section_name.replace("_", " ").title(),
                    "score": section_data["score"],
                    "color": self._get_score_color(section_data["score"])
                })
                
        analytics["visualization_data"] = visualization_data
        return analytics
    
    def _add_color_codes_to_scores(self, analysis_results):
        """Add color codes to scores for UI display"""
        # Add color to overall scores
        if "overall" in analysis_results:
            overall = analysis_results["overall"]
            if "score" in overall:
                overall["score_color"] = self._get_score_color(overall["score"])
            if "impact_score" in overall:
                overall["impact_score_color"] = self._get_score_color(overall["impact_score"])
            if "clarity_score" in overall:
                overall["clarity_score_color"] = self._get_score_color(overall["clarity_score"])
            if "relevance_score" in overall:
                overall["relevance_score_color"] = self._get_score_color(overall["relevance_score"])
            if "visual_appeal_score" in overall:
                overall["visual_appeal_score_color"] = self._get_score_color(overall["visual_appeal_score"])
                
            # Add priorities to top recommended changes
            if "top_recommended_changes" in overall and isinstance(overall["top_recommended_changes"], list):
                enhanced_changes = []
                for i, change in enumerate(overall["top_recommended_changes"]):
                    if isinstance(change, str):
                        priority = "high" if i < 2 else ("medium" if i < 4 else "low")
                        enhanced_changes.append({
                            "text": change,
                            "priority": priority,
                            "priority_color": self._get_priority_color(priority)
                        })
                    else:
                        enhanced_changes.append(change)
                overall["top_recommended_changes"] = enhanced_changes
        
        # Add color to ATS compatibility score
        if "ats_compatibility" in analysis_results and "ats_score" in analysis_results["ats_compatibility"]:
            analysis_results["ats_compatibility"]["ats_score_color"] = self._get_score_color(
                analysis_results["ats_compatibility"]["ats_score"]
            )
            
        # Add color to section scores
        for section_name, section_data in analysis_results.get("sections", {}).items():
            if "score" in section_data:
                section_data["score_color"] = self._get_score_color(section_data["score"])
    
    def _get_score_color(self, score):
        """Get appropriate color based on score for UI display"""
        if score >= 90:
            return "#27ae60"  # Green
        elif score >= 80:
            return "#2ecc71"  # Light Green
        elif score >= 70:
            return "#f1c40f"  # Yellow
        elif score >= 60:
            return "#e67e22"  # Orange
        else:
            return "#e74c3c"  # Red
            
    def _get_priority_color(self, priority):
        """Get appropriate color based on priority for UI display"""
        if priority.lower() == "high":
            return "#e74c3c"  # Red
        elif priority.lower() == "medium":
            return "#f1c40f"  # Yellow
        else:
            return "#2ecc71"  # Green

# Usage example (for testing)
if __name__ == "__main__":
    # Sample data for testing
    sample_parsed_resume = {
        "raw_content": "John Doe\nSoftware Engineer\n\nExperience:\nABC Company - Senior Developer (2020-Present)\nXYZ Corp - Junior Developer (2018-2020)\n\nEducation:\nBS Computer Science, University of Example (2018)",
        "metadata": {"filename": "sample_resume.pdf"},
        "sections": {
            "experience": ["ABC Company - Senior Developer (2020-Present)", "XYZ Corp - Junior Developer (2018-2020)"],
            "education": ["BS Computer Science, University of Example (2018)"],
            "skills": ["Python", "JavaScript", "React", "Node.js"]
        }
    }
    
    sample_job_description = "Looking for a Senior Software Engineer with 3+ years of experience in Python and JavaScript. React experience preferred."
    
    # Initialize analyzer and run test analysis
    analyzer = ResumeAnalyzer()
    result = analyzer.analyze_full_resume(sample_parsed_resume, sample_job_description)
    
    # Print results (for testing)
    print(json.dumps(result, indent=2))