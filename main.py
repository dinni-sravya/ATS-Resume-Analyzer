import os
import PyPDF2
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from google import genai
from google.genai import types

# ==============================
# CONFIGURATION
# ==============================
app = Flask(__name__)
CORS(app)  # Enable CORS

# Create uploads folder if it doesn't exist
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Initialize Gemini Client
# RECOMMENDATION: Use os.environ.get("GEMINI_API_KEY") for security
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyD1afIYywGsJsqhTgQP1LlOFogI-bkZs4U")
client = genai.Client(api_key=API_KEY)

# ==============================
# ROUTES
# ==============================

@app.route("/")
def home():
    """Serves the frontend HTML."""
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """Handles the Resume Analysis request."""
    if "resume" not in request.files:
        return jsonify({"error": "Resume PDF is required"}), 400

    resume_file = request.files["resume"]
    jd_text = request.form.get("job_description")

    if not jd_text:
        return jsonify({"error": "Job description is required"}), 400

    try:
        # 1. Save PDF locally
        pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], resume_file.filename)
        resume_file.save(pdf_path)

        # 2. Extract Text from PDF
        resume_text = extract_text_from_pdf(pdf_path)

        # 3. AI Processing
        parsed_resume = parse_resume(resume_text)
        parsed_jd = parse_job_description(jd_text)
        ats_result = ats_match(parsed_resume, parsed_jd)

        # Cleanup: Optional - remove uploaded file after processing
        # os.remove(pdf_path)

        return jsonify({
            "parsed_resume": parsed_resume,
            "parsed_job_description": parsed_jd,
            "ats_result": ats_result
        })

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

# ==============================
# HELPER FUNCTIONS
# ==============================

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def parse_resume(resume_text):
    prompt = f"""
    You are an expert Resume Parser. 
    Please analyze the following resume text and extract the key details in a clean format.

    Resume Text:
    {resume_text}

    Extract:
    - Candidate Name (if found)
    - Top 5 Technical Skills
    - Experience Summary (2 sentences max)
    - Education Summary
    
    Return the output in clean text format suitable for display.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error parsing resume: {str(e)}"

def parse_job_description(jd_text):
    prompt = f"""
    You are an expert HR Assistant.
    Analyze the following Job Description.

    Job Description:
    {jd_text}

    Extract:
    - Key Technical Requirements
    - Core Responsibilities
    - "Nice to have" skills
    
    Return the output in clean text format suitable for display.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error parsing JD: {str(e)}"

def ats_match(parsed_resume, parsed_jd):
    prompt = f"""
    You are an advanced Applicant Tracking System (ATS) Scanner.
    
    Task: Compare the parsed resume against the job description.

    Parsed Resume:
    {parsed_resume}

    Parsed Job Description:
    {parsed_jd}

    Output Requirements (Strict Format):
    1. Match Percentage: (Just the number followed by %)
    2. Matching Skills: (List key matches)
    3. Missing Skills: (List critical gaps)
    4. Verdict: (Short summary)
    5. Improvement Suggestions: (3 actionable tips)

    Make the tone professional and constructive.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error performing ATS match: {str(e)}"

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    app.run(debug=True, port=8000)