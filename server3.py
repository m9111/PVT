import os
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
import io
import asyncio
import openai
import uuid
from fastapi import UploadFile
from fastapi.responses import JSONResponse
import subprocess
import logging
import re
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
# Updated imports for memory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
# Updated imports for chains
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import aiohttp
from langchain.chains.question_answering import load_qa_chain
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.messages import get_buffer_string
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import librosa
import numpy as np

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 300))
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4-turbo')
RETRIEVAL_K = int(os.getenv('RETRIEVAL_K', 15))
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.3))

# Counter for Whisper API requests and lock for thread safety
whisper_request_counter = 0
whisper_counter_lock = threading.Lock()

DEFAULT_SYSTEM_MESSAGE = """You are UBIK AI, a helpful AI assistant with access to knowledge about UBIK solutions and its subsidiaries.

INSTRUCTIONS:
MAIN- DONT YAP, FOCUS ON QUESTION AND GIVE FOCUSED ANSWER , FOR EXAMPLE - THANKYOU,THEN REPLY FOR THANKYOU DONT LOOK FOR HISTORY AND YAP
1. Answer based ONLY on the provided context. If the question isn't covered in the context, say "I don't have information about that in my knowledge base." Do not make up information.
2. You can respond to greetings or general questions about UBIK and its subsidiaries.
3. Keep responses under 50 words unless detailed explanations are necessary.
4. CRITICAL: You MUST detect and correct ALL types of spelling errors in user messages, no matter how small the error. Be extremely vigilant about this.
5. Never discuss patient-related information.
6. Remember: U in UBIK stands for Utsav Khakhar, B for Bhavini Khakhar, I for Ilesh Khakhar, K for Khakhar.
7. For ANY typos or misspellings, even simple ones, you MUST use this format at the beginning of your answer: "[CORRECTION: original_text → correct_text]" - For example: "[CORRECTION: Utsav Gokhar → Utsav Khakhar]" and then continue with your answer.
8. Common misspellings to watch for: "ubik" might be "ubeek", "ubiik", "youbik"; "ethiglo" might appear as "ethiglow", "ethigloo", "ethi glo".
9. Even for minor typos like "prodacts" instead of "products" or "tehnology" instead of "technology", you MUST apply the correction format.
10. If multiple words are misspelled, provide the correction for the most important term first.
11. DONT ANSWER IN PARAGRAPH , GIVE SOME SPACES TOO , MAKE IT LINE BY LINE ALSO ADD SOME EMOJIS IF U WANT U CAN ADD AFTER EVERY 2 LINES OR 1 LINE, JUST FOCUS TO MAKE THING SENSIBLE ENOUGH
12. CRITICAL: IF YOU ARE GIVING NAME FOR EXAMPLE Mr. ILESH , THEN DONT WRITE "." AFTER Mr
13. critical: if you are listing products then dont use format of 1. 2. 3. , use 1) 2) 3) and after every product or listing u will use "."
EXAMPLES:

Question: "Wat r tha prodacts of Ubik?"
Good response: "[CORRECTION: Wat r tha prodacts → What are the products] [CORRECTION: Ubik → UBIK] UBIK's main products include..."

Question: "Tell me about Utsav Gokhar"
Good response: "[CORRECTION: Utsav Gokhar → Utsav Khakhar] Utsav Khakhar is the U in UBIK..."

Question: "How do I perform brain surgery?"
Good response: "I don't have information about that in my knowledge base. UBIK AI is focused on providing information about UBIK solutions and its subsidiaries."

Question: "wat iz ethi glo?"
Good response: "[CORRECTION: wat iz → what is] [CORRECTION: ethi glo → EthiGlo] EthiGlo is a product from UBIK Solutions that..."
"""

logging.basicConfig(
    filename='app_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=3)

global_vectorstore = None
vectorstore_lock = threading.Lock()

user_sessions = {}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def log_event(event_type: str, details: str = "", user_id: str = "Unknown"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"{timestamp} | {event_type} | UserID: {user_id} | {details}\n"
    with open("app_logs.txt", "a", encoding="utf-8") as f:
        f.write(log_line)

def get_user_state(user_id: str):
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            'memory': ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True
            ),
            'chain': None,
            'history': [],
            'system_message': DEFAULT_SYSTEM_MESSAGE
        }
        log_event("UserJoined", "New user state created.", user_id=user_id)
    return user_sessions[user_id]

@lru_cache(maxsize=128)
def get_pdf_text(pdf_path: str) -> str:
    try:
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return ""
            
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"Error reading {pdf_path}: {e}")
        return ""

def get_text_chunks(text: str):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # Add metadata to track chunk numbers for debugging
    logger.info(f"Created {len(chunks)} text chunks from PDFs")
    return chunks

# Function to reset the lru_cache for Whisper API
def reset_whisper_cache():
    """Reset the lru_cache for functions related to speech recognition"""
    handle_speech_to_text.cache_clear()
    logger.info("Whisper API cache has been reset")

# Add cache decorator to speech_to_text function
@lru_cache(maxsize=20)  # Cache up to 20 recent audio transcriptions
async def handle_speech_to_text(file: UploadFile):
    """
    Handle speech to text conversion using OpenAI's Whisper API.
    First checks if the audio file contains significant sound before processing.
   
    Args:
        file (UploadFile): The uploaded audio file
       
    Returns:
        JSON response with transcription or error
    """
    import os
    import tempfile
    import asyncio
    import aiohttp
    from fastapi import UploadFile
    from fastapi.responses import JSONResponse
    import logging
    import librosa
    import numpy as np
    
    global whisper_request_counter
    
    # Setup logging
    logger = logging.getLogger(__name__)
    
    # Custom words for better recognition
    CUSTOM_WORDS = [
        "ethiglo",
        "ethinext",
        "ubik",
        "sisonext",
        "retik",
        # Add your own specialized terms here
    ]
   
    # OpenAI API configuration
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set")
        return JSONResponse(
            status_code=500,
            content={"error": "OpenAI API key not configured."}
        )
   
    WHISPER_MODEL = "whisper-1"  # OpenAI's Whisper model identifier
   
    logger.info(f"Starting transcription for file: {file.filename}")
    if file is None:
        logger.error("No file provided")
        return JSONResponse(
            status_code=400,
            content={"error": "No file provided."}
        )
   
    temp_file = None
    try:
        # Read the file content
        file_content = await file.read()
       
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.write(file_content)
        temp_file.close()
       
        logger.info(f"Saved audio to temporary file: {temp_file.name}")
        
        # Analyze audio for significant sound
        try:
            # Load audio file with librosa
            y, sr = librosa.load(temp_file.name, sr=None)
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=y)[0]
            
            # Check if there's significant sound
            # A higher threshold means more sound is needed to be considered "significant"
            threshold = 0.01  # Adjust based on testing
            
            if np.mean(rms) < threshold:
                logger.info(f"No significant sound detected in file: {file.filename}")
                return {"status": "success", "text": " "}
            
            logger.info(f"Significant sound detected in file: {file.filename}, proceeding with transcription")
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            # Continue with transcription even if analysis fails
       
        # Prepare the custom prompt
        custom_prompt = ", ".join(CUSTOM_WORDS) if CUSTOM_WORDS else ""
        prompt = f"Please see the best suitable transcription, if available from custom words: {custom_prompt}. "
       
        # Function to call OpenAI API
        async def call_openai_api():
            async with aiohttp.ClientSession() as session:
                api_url = "https://api.openai.com/v1/audio/transcriptions"
                
                headers = {
                    "Authorization": f"Bearer {OPENAI_API_KEY}"
                }
                
                # Prepare form data
                form_data = aiohttp.FormData()
                form_data.add_field("file", 
                                   open(temp_file.name, "rb"),
                                   filename=os.path.basename(temp_file.name),
                                   content_type="audio/wav")
                form_data.add_field("model", WHISPER_MODEL)
                form_data.add_field("language", "en")  # Specify language if known
                form_data.add_field("prompt", prompt)
                
                async with session.post(api_url, headers=headers, data=form_data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error: {error_text}")
                        return None
                    
                    result = await response.json()
                    return result.get("text", "")
       
        # Call the OpenAI API
        transcript = await call_openai_api()
        
        if transcript is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to get transcription from OpenAI."}
            )
       
        logger.info("Successfully transcribed audio using OpenAI Whisper API")
        
        # Increment the request counter with thread safety
        with whisper_counter_lock:
            global whisper_request_counter
            whisper_request_counter += 1
            current_count = whisper_request_counter
            logger.info(f"Whisper API request count: {current_count}")
            
            # Reset cache after every 2 requests
            if current_count % 2 == 0:
                # Use executor to reset cache asynchronously without blocking
                executor.submit(reset_whisper_cache)
                logger.info("Scheduled cache reset after 2 requests")
        
        return {"status": "success", "text": transcript}
       
    except Exception as e:
        logger.exception("Unexpected error occurred during transcription")
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred during transcription: {str(e)}"}
        )
    finally:
        # Clean up the temporary file
        if temp_file and os.path.exists(temp_file.name):
            os.remove(temp_file.name)
            logger.info(f"Removed temporary file: {temp_file.name}")

# Make the function cache clearable
handle_speech_to_text.cache_clear = lambda: None

@app.post("/speech_to_text")
async def speech_to_text_endpoint(file: UploadFile = File(...)):
    global whisper_request_counter
    response = await handle_speech_to_text(file)
    
    # Check if we should force a cache reset before the 3rd request
    with whisper_counter_lock:
        if whisper_request_counter == 2:
            # Reset cache immediately before proceeding to the 3rd request
            reset_whisper_cache()
            logger.info("Cache reset forced before 3rd request")
    
    return response

@app.post("/reset_whisper_cache")
async def reset_whisper_cache_endpoint():
    """Endpoint to manually reset the Whisper API cache"""
    try:
        reset_whisper_cache()
        with whisper_counter_lock:
            global whisper_request_counter
            whisper_request_counter = 0
            logger.info("Whisper request counter reset to 0")
        return {"status": "success", "message": "Whisper API cache reset successfully"}
    except Exception as e:
        logger.exception(f"Error resetting Whisper cache: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error resetting Whisper cache: {str(e)}"}
        )

def initialize_global_vectorstore():
    global global_vectorstore
    
    try:
        with vectorstore_lock:
            if global_vectorstore is not None:
                return True, "[SYSTEM MESSAGE] Vectorstore is already initialized."

            # Check if 'data' directory exists
            if not os.path.exists('data'):
                os.makedirs('data')
                logger.info("Created 'data' directory")
            
            pdf_paths = [
                os.path.join('data', "Ilesh Sir (IK) - Words.pdf"),
                os.path.join('data', "UBIK SOLUTION.pdf"),
                os.path.join('data', "illesh3.pdf"),
                os.path.join('data', "website-data-ik.pdf"),
                os.path.join('data', "prods1.pdf"),
                os.path.join('data', "summary-gem.pdf"),
                os.path.join('data', "summary-gem2.pdf")
            ]

            # Check if the PDF files exist and log which ones are missing
            missing_pdfs = []
            for path in pdf_paths:
                if not os.path.exists(path):
                    missing_pdfs.append(path)
                    logger.error(f"PDF file not found: {path}")
            
            if missing_pdfs:
                logger.error(f"The following PDF files are missing: {missing_pdfs}")
                return False, f"Missing PDF files: {', '.join(missing_pdfs)}"

            combined_text = ""
            for path in pdf_paths:
                if path in missing_pdfs:
                    continue
                    
                pdf_text = get_pdf_text(path)
                if pdf_text:
                    # Add a clear marker for each document source to help with context
                    combined_text += f"\n\n### DOCUMENT: {os.path.basename(path)} ###\n\n" + pdf_text + " "
                    logger.info(f"Added content from {path} to combined text")
                else:
                    logger.warning(f"No text extracted from {path}")

            if not combined_text.strip():
                logger.error("No text could be extracted from any of the PDFs.")
                return False, "No text could be extracted from the PDFs."

            text_chunks = get_text_chunks(combined_text)
            
            if not text_chunks:
                logger.error("No text chunks were created from the PDFs.")
                return False, "No text chunks were created from the PDFs."
            
            logger.info(f"Created {len(text_chunks)} text chunks from PDFs")
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY environment variable not set")
                return False, "OpenAI API key not configured."
            
            embeddings = OpenAIEmbeddings(
                api_key=api_key,
                model="text-embedding-3-small"  # Use newer embedding model for better performance
            )
            
            # Create vectorstore with additional parameters for better retrieval
            global_vectorstore = FAISS.from_texts(
                texts=text_chunks,
                embedding=embeddings,
                metadatas=[{"chunk": i, "source": "pdf_collection"} for i in range(len(text_chunks))]
            )
            logger.info("Vectorstore has been created with OpenAI embeddings.")
        return True, "[SYSTEM MESSAGE] Vectorstore was created successfully."
    
    except Exception as e:
        logger.exception(f"Error initializing vectorstore: {e}")
        return False, f"Error initializing vectorstore: {str(e)}"

def handle_userinput(user_question: str, user_id: str):
    try:
        user_state = get_user_state(user_id)
        conversation_chain = user_state['chain']
        if not conversation_chain:
            success, message = create_or_refresh_user_chain(user_id)
            if not success:
                logger.error(f"Failed to create conversation chain: {message}")
                return {'text': f"I'm having trouble processing your question. Please try again later. Error: {message}"}
            conversation_chain = user_state['chain']
            if not conversation_chain:
                logger.error("Conversation chain is still None after refresh attempt")
                return {'text': "I'm having trouble connecting to my knowledge base. Please try again later."}

        # Log the question for debugging
        logger.info(f"Processing question from user {user_id}: {user_question}")
        
        # Try using a timeout to prevent hanging
        try:
            # Fixed: Use invoke instead of __call__
            response = conversation_chain.invoke({
                "question": user_question
            })
            
            if not response:
                logger.error(f"Empty response from conversation chain for user {user_id}")
                return {'text': "I received your question but couldn't generate a response. Please try again."}
                
            answer = response.get('answer', '').strip()
            if not answer:
                logger.error(f"Empty answer in response: {response}")
                return {'text': "I processed your question but couldn't find a suitable answer. Please try rephrasing."}
                
            transformed_answer = transform_ai_text(answer)
    
            user_state['history'].append((user_question, transformed_answer))
            log_event("PromptSentToGPT", f"Original Prompt: {user_question}", user_id=user_id)
            log_event("UserQuestion", f"Q: {user_question}", user_id=user_id)
            log_event("AIAnswer", f"A: {transformed_answer}", user_id=user_id)
    
            return {'text': transformed_answer}
        except Exception as chain_error:
            logger.exception(f"Error in conversation chain: {chain_error}")
            
            # Fallback mechanism - try a direct response if the chain fails
            try:
                logger.info(f"Attempting fallback response for user {user_id}")
                api_key = os.getenv("OPENAI_API_KEY")
                llm = ChatOpenAI(model=MODEL_NAME, temperature=0.2, api_key=api_key)
                
                # Get last few messages from memory for context
                memory_messages = []
                if user_state['memory']:
                    memory_messages = user_state['memory'].chat_memory.messages[-4:] if hasattr(user_state['memory'], 'chat_memory') else []
                
                # Create a simple fallback prompt
                fallback_prompt = f"""
                {user_state['system_message']}
                
                The user asked: {user_question}
                
                Please provide a helpful response based on what you know about UBIK and its subsidiaries.
                If you don't have specific information, provide a general helpful response.
                """
                
                fallback_response = llm.invoke(fallback_prompt)
                fallback_answer = fallback_response.content if hasattr(fallback_response, 'content') else str(fallback_response)
                
                transformed_fallback = transform_ai_text(fallback_answer)
                user_state['history'].append((user_question, transformed_fallback))
                
                logger.info(f"Successfully generated fallback response for user {user_id}")
                return {'text': transformed_fallback}
            except Exception as fallback_error:
                logger.exception(f"Fallback response also failed: {fallback_error}")
                return {'text': "I'm having trouble answering your question right now. Please try again with a different question."}
    
    except Exception as e:
        logger.exception(f"Error handling user input: {e}")
        return {'text': "I encountered an error while processing your question. Please try again later."}

def transform_ai_text(text):
    try:
        # Temporarily protect correction markers from other transformations
        protected_text = text
        correction_matches = []
        
        # Find and protect all correction markers
        correction_pattern = r'\[CORRECTION:[^\]]*\]'
        for match in re.finditer(correction_pattern, text):
            correction_matches.append(match.group(0))
            # Replace with a placeholder
            placeholder = f"__CORRECTION_PLACEHOLDER_{len(correction_matches)-1}__"
            protected_text = protected_text.replace(match.group(0), placeholder)
        
        # Apply normal transformations to the protected text
        transformed_text = re.sub(r'(\d)\.(\d)', r'\1point\2', protected_text)
        transformed_text = transformed_text.replace('*', ' ')
        
        # Restore correction markers
        for i, correction in enumerate(correction_matches):
            placeholder = f"__CORRECTION_PLACEHOLDER_{i}__"
            transformed_text = transformed_text.replace(placeholder, correction)
        
        return transformed_text
    
    except Exception as e:
        logger.exception(f"Error transforming AI text: {e}")
        return text  # Return original text on error

def create_or_refresh_user_chain(user_id: str):
    try:
        user_state = get_user_state(user_id)
        
        # Check if vectorstore is initialized
        if global_vectorstore is None:
            success, message = initialize_global_vectorstore()
            if not success:
                logger.error(f"Failed to initialize vectorstore: {message}")
                return False, message
        
        # Create the chain if it doesn't exist
        if user_state['chain'] is None:
            logger.info(f"Creating new conversation chain for user {user_id}")
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY environment variable not set")
                return False, "OpenAI API key not configured."
            
            chat_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.2, request_timeout=60)

            # Updated condense question prompt with logic to detect standalone questions
            condense_template = """Given the following conversation and a follow up question, decide if the follow up question should be processed as a standalone question.

            First, determine if the follow up input is a simple greeting, acknowledgment, or simple phrase (like "hi", "hello", "thanks", "thank you", "ok", etc.) that should be treated independently.
            
            Chat History:
            {chat_history}
            
            Follow Up Input: {question}
            
            IMPORTANT: Always check for typos, misspellings, and incorrect capitalization in the follow-up input. This is a CRITICAL PRIORITY.

            If you detect any spelling errors, you must address them with a correction in your answer using the exact format: "[CORRECTION: misspelled_text → correct_text]".
            
            After identifying a correction, you MUST work with the corrected version of the term. For example, if the follow-up input is "Tell me about ubik" and you correct it to "UBIK", you should process the question as "Tell me about UBIK".
            
            If the follow up input is a simple standalone phrase or greeting, output it exactly as is.
            Otherwise, rephrase it to be a standalone question that captures the context from the conversation.
            
            Handle spelling errors and typos in the follow up input - try to understand what the user meant even if there are significant spelling mistakes.
            
            Processed question:"""
            
            CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

            qa_template = f"""
            {user_state['system_message']}

            Context: {{context}}
            
            Question: {{question}}
            
            Remember: 
            - If the question is a simple greeting like "hi", "hello", "thanks", or similar, respond ONLY to that greeting without referencing previous conversation topics. DONT YAP, FOCUS ON QUESTION AND GIVE FOCUSED ANSWER.
            - You MUST detect and correct ALL spelling errors, typos, or incorrect capitalizations in the question. ALWAYS use the format: "[CORRECTION: misspelled_text → correct_text]"
            - After identifying a correction, use the CORRECTED version of the term when searching your knowledge. Do NOT say "I don't have information" immediately after correcting a term.
            - CRITICAL: When you correct a spelling error, you MUST then answer the question using the correctly spelled term. Never respond with "I don't have information" immediately after making a correction unless you truly don't have information about the correctly spelled term.
            - Even for simple or minor spelling mistakes like "tehnology" instead of "technology", you MUST include the correction.
            - Pay special attention to company names, product names, and proper nouns that might be misspelled.
            - Examine every word in the question for possible errors, even if it seems mostly correct.
            - Common misspellings you may encounter: "ubik" as "ubeek"/"youbik", "ethiglo" as "ethiglow"/"ethi glo", etc.
            - If you don't find exact information in the context, try to provide a related answer that might be helpful.
            - If the context doesn't have enough information, try to give a general answer based on what you know about UBIK and its subsidiaries.
            - Be creative in your responses while maintaining accuracy - connect related information when appropriate.
            
            Answer: """
            
            QA_PROMPT = PromptTemplate(
                template=qa_template,
                input_variables=["context", "question"]
            )

            # Get the retriever from the global vectorstore
            retriever = global_vectorstore.as_retriever(
                search_type="mmr",  # Use Maximum Marginal Relevance for more diverse results
                search_kwargs={
                    "k": RETRIEVAL_K,
                    "score_threshold": SIMILARITY_THRESHOLD,
                    "fetch_k": RETRIEVAL_K * 2  # Fetch more candidates for MMR to choose from
                }
            )

            # Updated to use the new ConversationalRetrievalChain approach
            user_state['chain'] = ConversationalRetrievalChain.from_llm(
                llm=chat_llm,
                retriever=retriever,
                memory=user_state['memory'],
                condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                combine_docs_chain_kwargs={'prompt': QA_PROMPT}
            )
            
            logger.info(f"Successfully created conversation chain for user {user_id}")
            return True, "Conversation chain created successfully."
        else:
            logger.info(f"Conversation chain already exists for user {user_id}")
            return True, "Conversation chain already exists."
    
    except Exception as e:
        logger.exception(f"Error creating conversation chain: {e}")
        return False, f"Error creating conversation chain: {str(e)}"

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Starting application and initializing vectorstore...")
        success, message = initialize_global_vectorstore()
        if not success:
            logger.error(f"Failed to initialize vectorstore: {message}")
        else:
            logger.info("Vectorstore initialized successfully on startup")
    except Exception as e:
        logger.exception(f"Error during startup: {e}")

@app.get("/")
async def hello_root():
    return {
        "message": "Hello from FastAPI server. Use the endpoints to process data or ask questions.",
        "status": "operational"
    }

@app.post("/refresh_chain")
async def refresh_chain(request: Request):
    try:
        data = await request.json()
        if "user_id" not in data:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing user_id."}
            )

        user_id = data["user_id"]
        
        # First, ensure vectorstore is initialized
        if global_vectorstore is None:
            vsuccess, vmessage = initialize_global_vectorstore()
            if not vsuccess:
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Could not initialize vectorstore: {vmessage}"}
                )
        
        # Then try to create/refresh the chain
        success, message = create_or_refresh_user_chain(user_id)
        status = 'success' if success else 'error'
        
        if not success:
            logger.error(f"Failed to refresh chain for user {user_id}: {message}")
            return JSONResponse(
                status_code=500,
                content={"status": status, "error": message}
            )
            
        return {"status": status, "message": message}
    
    except Exception as e:
        logger.exception(f"Error in refresh_chain endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred: {str(e)}"}
        )

@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        logger.exception(f"Invalid JSON payload: {e}")
        return JSONResponse(status_code=400, content={"error": "Invalid JSON payload."})

    if "user_id" not in data or "question" not in data:
        return JSONResponse(status_code=400, content={"error": "Missing user_id or question."})

    user_id = data["user_id"]
    user_question = data["question"]
    FORWARD_ENDPOINT = os.getenv("FORWARD_ENDPOINT", "https://win-ldm7g6fmsik.tail5e2acc.ts.net/receive")

    # Check if vectorstore is initialized
    if global_vectorstore is None:
        vsuccess, vmessage = initialize_global_vectorstore()
        if not vsuccess:
            logger.error(f"Could not initialize vectorstore: {vmessage}")
            return {"status": "error", "message": f"Could not initialize knowledge base: {vmessage}"}

    # Ensure user has a conversation chain
    if user_id not in user_sessions or user_sessions[user_id]['chain'] is None:
        success, message = create_or_refresh_user_chain(user_id)
        if not success:
            logger.error(f"Failed to create conversation chain: {message}")
            return {"status": "error", "message": f"Failed to initialize conversation: {message}"}

    # Process the question
    answer = handle_userinput(user_question, user_id)
    if not answer or 'text' not in answer:
        logger.error(f"No valid answer returned for user {user_id}")
        return {"status": "error", "message": "Unable to process your question. Please try again."}

    # Forward the response if needed
    forwarding_status = "disabled"
    try:
        if FORWARD_ENDPOINT:
            payload = {
                "text": answer['text']
            }
            
            forward_response = requests.post(FORWARD_ENDPOINT, json=payload)
            forwarding_status = "success" if forward_response.status_code == 200 else "failed"
            
            if forward_response.status_code != 200:
                logger.error(f"Forward request failed with status {forward_response.status_code}: {forward_response.text}")
            else:
                logger.info(f"Response forwarding {forwarding_status}")
    except Exception as e:
        logger.error(f"Error forwarding response: {e}")
        forwarding_status = "failed"

    prompt_sent = {'question': user_question}
    return {
        "status": "success", 
        "data": answer, 
        "prompt": prompt_sent,
        "forwarding_status": forwarding_status
    }

@app.post("/set_system_message")
async def set_system_message(request: Request):
    try:
        data = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON."})

    user_id = data.get("user_id")
    system_message = data.get("system_message")

    if not user_id or not system_message:
        return JSONResponse(
            status_code=400, 
            content={"error": "Missing user_id or system_message."}
        )

    user_state = get_user_state(user_id)
    user_state['system_message'] = system_message
    
    # Clear the existing chain to force recreation with new system message
    user_state['chain'] = None
    
    # Create a new chain with the updated system message
    success, message = create_or_refresh_user_chain(user_id)
    
    log_event(
        "SystemMessageUpdated",
        f"System message updated to: {system_message}",
        user_id=user_id
    )
    
    return {
        "status": "success" if success else "error",
        "message": f"System message updated. {message}"
    }

@app.get("/get_system_message")
async def get_system_message(user_id: str):
    if not user_id:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing user_id."}
        )

    user_state = get_user_state(user_id)
    return {
        "status": "success",
        "system_message": user_state['system_message']
    }

@app.post("/clear_history")
async def clear_history(request: Request):
    try:
        data = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON."})

@app.post("/logout")
async def logout(request: Request):
    try:
        data = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON."})

    user_id = data.get("user_id", None)
    if not user_id:
        return JSONResponse(status_code=400, content={"error": "Missing user_id."})

    if user_id in user_sessions:
        del user_sessions[user_id]
    return {"status": "success", "message": "User session cleared."}

GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY")

@app.post("/text_to_speech")
async def text_to_speech_api(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        voice_name = data.get("voice", "en-US-Wavenet-D")
        language_code = data.get("language_code", "en-US")
        speaking_rate = data.get("speaking_rate", 1.0)

        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing 'text' in request."}
            )

        url = "https://texttospeech.googleapis.com/v1/text:synthesize?key=" + GOOGLE_TTS_API_KEY

        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": language_code,
                "name": voice_name,
            },
            "audioConfig": {
                "audioEncoding": "MP3",
                "speakingRate": speaking_rate,
            },
        }

        response = requests.post(url, json=payload)

        if response.status_code == 200:
            audio_content = response.json().get("audioContent", None)
            if not audio_content:
                return JSONResponse(
                    status_code=500,
                    content={"error": "No audio content received from TTS API."}
                )

            audio_path = os.path.join("data", f"tts_output_{uuid.uuid4().hex}.mp3")
            with open(audio_path, "wb") as audio_file:
                audio_file.write(base64.b64decode(audio_content))

            return {"status": "success", "audio_url": audio_path}
        else:
            return JSONResponse(
                status_code=response.status_code,
                content={"error": response.text}
            )

    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "An internal error occurred during text-to-speech synthesis."}
        )

@app.post("/tts_proxy")
async def tts_proxy(request: Request):
    """
    Proxy endpoint for Google Text-to-Speech API used by TalkingHead
    This keeps the API key secure on the server side
    """
    try:
        # Extract data from request
        data = await request.json()
        logger.info(f"TTS Proxy request received: {data}")
        
        # Add the API key directly to the URL query parameter as expected by Google's API
        url = f"https://eu-texttospeech.googleapis.com/v1beta1/text:synthesize?key={GOOGLE_TTS_API_KEY}"
        
        # Forward the request to Google
        headers = {
            "Content-Type": "application/json"
        }
        
        # Forward the request to Google API
        response = requests.post(url, headers=headers, json=data)
        
        # Log the response status and headers
        logger.info(f"TTS Proxy response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"TTS API Error: {response.text}")
            return JSONResponse(
                status_code=response.status_code,
                content={"error": response.text}
            )
        
        # Return the response directly
        return response.json()
        
    except Exception as e:
        logger.error(f"Error in TTS proxy: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing TTS request: {str(e)}"}
        )

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=9000)