from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import os
import os
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import uuid
import time
import logging
import threading
from datetime import datetime
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import base64
import requests
import speech_recognition as sr
import os
import uuid
from fastapi import UploadFile
from fastapi.responses import JSONResponse
import subprocess
import logging
import uuid
import time
import logging
import threading
from datetime import datetime
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import base64
import requests
import speech_recognition as sr
import os
import uuid
from fastapi import UploadFile
from fastapi.responses import JSONResponse
import subprocess
import logging
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import json
import speech_recognition as sr
import base64
import io
from fastapi import FastAPI, File, UploadFile, Form
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
from pydub import AudioSegment
import io
import tempfile
import os
# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage class
class QuizStorage:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.user_sessions: Dict[str, Dict] = {}

quiz_storage = QuizStorage()

# Pydantic models
class QuizResponse(BaseModel):
    question_id: int
    answer: str

class VoiceInput(BaseModel):
    audio_data: str
    user_id: str

class QuizResult(BaseModel):
    total_score: float
    max_score: float
    feedback: List[Dict[str, str]]
    improvement_areas: List[str]

def generate_unique_questions() -> List[str]:
    """Generate unique questions about UBIK Solutions products similar to the provided examples."""
    question_template = """
    I need you to generate 5 unique questions that customers or doctors might ask about UBIK Solutions products.
    The questions should match the style and format of these examples:

    1. Why should I support UBIK Solutions?
    2. Other companies are giving more benefits why should I write your products?
    3. UBIK is charging almost for everything.
    4. Your products are available online.
    5. What makes your products clinically superior to other companies' products?
    6. How are you ensuring the quality and consistency of your products?
    7. What kind of research or evidence supports your products' claims?
    8. Why is your company not more flexible in payment options or discounts?
    9. What initiatives are you taking to support newer doctors?
    10. What differentiates your training and support programs from other companies'?

    The questions should:
    - Address concerns about product quality, pricing, availability, or support
    - Focus on objections doctors might have about prescribing UBIK products
    - Include questions about competitive advantages over other companies
    - Include questions about company policies, services or support programs
    
    Return only a JSON array of 5 question strings, formatted as a valid JSON array.
    """
    
    try:
        # Use a simpler prompt-based approach rather than retrieval
        llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=OPENAI_API_KEY
        )
        
        # Generate questions using a direct prompt
        response = llm.predict(question_template)
        
        # Parse and validate response
        response = response.replace("'", '"').strip()
        if '[' in response and ']' in response:
            response = response[response.find('['):response.rfind(']')+1]
        questions = json.loads(response)
        
        return questions[:5]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Question generation failed: {str(e)}"
        )
logging.basicConfig(
    filename='app_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize PDFs at startup

@app.on_event("startup")
async def startup_event():
    pdf_dir = "datamn"
    try:
        # Load documents with larger chunks
        documents = []
        for pdf_file in os.listdir(pdf_dir):
            if pdf_file.endswith('.pdf'):
                loader = PyPDFLoader(os.path.join(pdf_dir, pdf_file))
                documents.extend(loader.load())
        
        # Optimize text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Increased chunk size
            chunk_overlap=100,  # Reduced overlap
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        
        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-ada-002"
        )
        quiz_storage.vector_store = FAISS.from_documents(splits, embeddings)
        
        # Initialize LLM with optimized settings
        llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=OPENAI_API_KEY
        )
        
        # Create optimized QA chain
        quiz_storage.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # Changed to simpler chain type
            retriever=quiz_storage.vector_store.as_retriever(
                search_kwargs={"k":5} # Reduced number of documents
            ),
            verbose=False  # Disabled verbose logging
        )
        
        # Initialize dedicated question generation chain
        question_prompt = PromptTemplate(
            template="""
            Based on the following context, generate 5 unique and challenging dermatology questions.
            Questions should:
            - Focus on different aspects of dermatology (diagnosis, treatment, pathology)
            - Test application of knowledge rather than recall
            - Be clinically relevant and practice-oriented
            
            Context: {context}
            
            Return only a JSON array of 5 question strings.
            """,
            input_variables=["context"]
        )
        
        quiz_storage.question_chain = LLMChain(
            llm=llm,
            prompt=question_prompt,
            verbose=False
        )
        
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        raise

@app.post("/speech-to-text")
async def speech_to_text(audio: UploadFile = File(...)):
    try:
        # Read the uploaded audio file
        audio_content = await audio.read()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(audio_content)
            temp_audio_path = temp_audio.name

        # Convert audio to WAV format
        audio = AudioSegment.from_file(temp_audio_path)
        audio.export(temp_audio_path, format="wav")

        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Perform recognition
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        # Clean up
        os.unlink(temp_audio_path)
        
        return {"text": text}
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/start-quiz/{user_id}")
async def start_quiz(user_id: str):
    if not quiz_storage.qa_chain:
        raise HTTPException(status_code=500, detail="Quiz system not properly initialized")
    
    questions = generate_unique_questions()
    
    quiz_storage.user_sessions[user_id] = {
        "current_question": 0,
        "answers": [],
        "scores": [],
        "feedback": [],
        "questions": questions
    }
    
    return get_next_question(user_id)

@app.post("/speech-to-text")
async def speech_to_text(audio: UploadFile = File(...)):
    try:
        audio_content = await audio.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(audio_content)
            temp_audio_path = temp_audio.name

        audio = AudioSegment.from_file(temp_audio_path)
        audio.export(temp_audio_path, format="wav")

        recognizer = sr.Recognizer()
        
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        os.unlink(temp_audio_path)
        
        return {"text": text}
    
    except Exception as e:
        return {"error": str(e)}



@app.post("/answer-question/{user_id}")
async def answer_question(user_id: str, response: QuizResponse):
    if user_id not in quiz_storage.user_sessions:
        raise HTTPException(status_code=404, detail="Quiz session not found")
    
    session = quiz_storage.user_sessions[user_id]
    
    if session["current_question"] >= 5:
        raise HTTPException(status_code=400, detail="Quiz already completed")
    
    current_question = session["questions"][session["current_question"]]
    
    # Get ideal answer for all questions, not just skipped ones
    ideal_answer = quiz_storage.qa_chain.run(
        f"Provide a concise ideal answer for: {current_question}"
    )
    
    # Handle skipped questions
    if response.answer == "SKIPPED":
        session["answers"].append("SKIPPED")
        session["scores"].append(0)
        session["feedback"].append({
            "question": current_question,
            "feedback": "Question was skipped",
            "improvement": "Study the provided ideal answer",
            "ideal_answer": ideal_answer
        })
        session["current_question"] += 1
        
        return get_next_question(user_id) if session["current_question"] < 5 else calculate_results(user_id)
    
    # Evaluate non-skipped answers with structured prompt
    try:
        eval_prompt = f"""
        You are an expert evaluator. Evaluate this answer carefully:
        
        Question: {current_question}
        Student Answer: {response.answer}
        
        Provide your evaluation in the following JSON format exactly:
        {{
            "score": <number between 0 and 1>,
            "feedback": "<specific feedback on the answer>",
            "improvement": "<one key area for improvement>"
        }}
        
        Ensure the response is valid JSON with these exact keys.
        """
        
        # Get raw response from QA chain
        eval_response = quiz_storage.qa_chain.run(eval_prompt)
        
        # Clean the response to ensure valid JSON
        eval_response = eval_response.strip()
        if eval_response.startswith('```json'):
            eval_response = eval_response[7:-3]  # Remove ```json and ``` if present
        elif eval_response.startswith('```'):
            eval_response = eval_response[3:-3]  # Remove ``` if present
            
        # Parse JSON with error handling
        try:
            eval_data = json.loads(eval_response)
            
            # Validate required keys
            required_keys = {'score', 'feedback', 'improvement'}
            if not all(key in eval_data for key in required_keys):
                raise ValueError("Missing required keys in evaluation response")
                
            # Validate score is a number between 0 and 1
            score = float(eval_data['score'])
            if not 0 <= score <= 1:
                score = max(0, min(1, score))  # Clamp between 0 and 1
                
        except json.JSONDecodeError as json_err:
            # Fallback evaluation if JSON parsing fails
            eval_data = {
                'score': 0.5,  # Default middle score
                'feedback': "Unable to parse evaluation properly. Please review with instructor.",
                'improvement': "System error in evaluation - please try again"
            }
            logging.error(f"JSON parsing error: {json_err}\nRaw response: {eval_response}")
        
        # Update session with evaluation results and include ideal answer for all questions
        session["answers"].append(response.answer)
        session["scores"].append(float(eval_data["score"]))
        session["feedback"].append({
            "question": current_question,
            "feedback": eval_data["feedback"],
            "improvement": eval_data["improvement"],
            "ideal_answer": ideal_answer  # Add ideal answer for all questions
        })
        session["current_question"] += 1
        
        # Return next question or final results
        if session["current_question"] >= 5:
            return calculate_results(user_id)
        else:
            return get_next_question(user_id)
            
    except Exception as e:
        logging.error(f"Error in answer evaluation: {str(e)}\nUser ID: {user_id}\nResponse: {response}")
        raise HTTPException(
            status_code=500,
            detail=f"Error evaluating answer: {str(e)}"
        )

@app.post("/process-voice")
async def process_voice(voice_input: VoiceInput):
    try:
        audio_bytes = base64.b64decode(voice_input.audio_data)
        audio_file = io.BytesIO(audio_bytes)
        
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return {"success": True, "text": text}
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing voice input: {str(e)}"
        )

def get_next_question(user_id: str) -> Dict:
    session = quiz_storage.user_sessions[user_id]
    return {
        "question_number": session["current_question"] + 1,
        "question_text": session["questions"][session["current_question"]]
    }


def calculate_results(user_id: str) -> QuizResult:
    session = quiz_storage.user_sessions[user_id]
    
    total_score = sum(session["scores"])
    
    return QuizResult(
        total_score=total_score,
        max_score=5.0,
        feedback=session["feedback"],
        improvement_areas=[fb["improvement"] for fb in session["feedback"]]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)