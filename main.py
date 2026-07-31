import os
import uuid
import shutil
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query, status, File, UploadFile, Request, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from bson import ObjectId
from bson.errors import InvalidId

# Configure Matplotlib to run headless (silently) to prevent GUI blocking on the server
import matplotlib
matplotlib.use('Agg')

# Configuration and Database
from config import logger, UPLOAD_DIR, MAX_FILE_SIZE_BYTES, ALLOWED_FILE_TYPES, HISTORICAL_PATIENT_COUNT
from database import client, patients_collection, reports_collection, historical_collection, chats_collection
from models import Patient, PatientUpdate, Chat

# AI Pipeline functions & new chat context function
from ai import (
    process_uploaded_pdfs, extract_patient_information, generate_visualizations,
    generate_historical_database, build_faiss_index, get_similarity_context,
    doctor1_chatbot, doctor2_chatbot, chat_with_patient_context
)

# ==========================================
# FASTAPI INITIALIZATION
# ==========================================
app = FastAPI(
    title="Hospital AI Copilot API",
    description="Backend API for Hospital AI Copilot",
    version="1.0.0"
)

# ==========================================
# CORS MIDDLEWARE CONFIGURATION
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ==========================================
# FORCE OPENAPI 3.0 SCHEMA
# (Swagger UI's file-picker widget doesn't render OpenAPI 3.1's
#  contentMediaType-based file schema; 3.0's format:binary works reliably)
# ==========================================
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema["openapi"] = "3.0.3"

    # Patch file-upload fields: Pydantic v2 emits "contentMediaType" for
    # UploadFile, which Swagger UI does not render as a file picker.
    # Rewrite it to the older "format": "binary" style, which it does.
    for schema_name, schema_def in schema.get("components", {}).get("schemas", {}).items():
        for prop_name, prop_def in schema_def.get("properties", {}).items():
            # Case 1: single file -> {"type": "string", "contentMediaType": "..."}
            if prop_def.get("contentMediaType"):
                prop_def.pop("contentMediaType", None)
                prop_def["type"] = "string"
                prop_def["format"] = "binary"
            # Case 2: list of files -> {"type": "array", "items": {"contentMediaType": "..."}}
            items = prop_def.get("items")
            if isinstance(items, dict) and items.get("contentMediaType"):
                items.pop("contentMediaType", None)
                items["type"] = "string"
                items["format"] = "binary"

    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi
# ==========================================
# GLOBAL EXCEPTION HANDLERS
# ==========================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Ensures HTTP exceptions return the standardized success/message/detail format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "message": "Request failed", "detail": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catches all unhandled exceptions, logs them safely, and returns a generic 500 error."""
    logger.error(f"Unhandled server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "Internal Server Error", "detail": "An unexpected error occurred."}
    )

# ==========================================
# GLOBAL AI STATE (LAZY INITIALIZATION)
# ==========================================
rag_embedder = None
rag_faiss_index = None
rag_historical_texts = []

def initialize_ai_rag():
    """Lazily initializes the FAISS index. Uses existing MongoDB records if available."""
    global rag_embedder, rag_faiss_index, rag_historical_texts
    if rag_faiss_index is None:
        logger.info("Initializing historical RAG database...")
        existing_count = historical_collection.count_documents({})
        if existing_count > 0:
            logger.info(f"Loaded {existing_count} existing historical patients from MongoDB.")
            records = list(historical_collection.find({}, {"Patient_Summary": 1, "_id": 0}))
            rag_historical_texts = [r["Patient_Summary"] for r in records if "Patient_Summary" in r]
        else:
            logger.info("Generating synthetic historical patients and saving to MongoDB...")
            historical_df = generate_historical_database(num_patients=HISTORICAL_PATIENT_COUNT)
            historical_collection.insert_many(historical_df.to_dict(orient="records"))
            rag_historical_texts = historical_df['Patient_Summary'].tolist()

        rag_embedder, rag_faiss_index = build_faiss_index(rag_historical_texts)
        logger.info("RAG Index successfully built.")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def serialize_doc(doc: dict) -> dict:
    """Converts MongoDB ObjectId into a string for JSON responses."""
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc

def create_upload_folder():
    """Ensures the upload directory exists."""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

def validate_file_size(file: UploadFile):
    """Validates file size against MAX_FILE_SIZE_BYTES limits."""
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File '{file.filename}' exceeds {MAX_FILE_SIZE_BYTES / (1024 * 1024)} MB limit."
        )

async def save_uploaded_files(files: List[UploadFile]) -> List[dict]:
    """Saves uploaded files securely with UUIDs into dated directories."""
    saved_files_info = []
    dated_upload_dir = os.path.join(UPLOAD_DIR, datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(dated_upload_dir):
        os.makedirs(dated_upload_dir)

    for file in files:
        unique_filename = f"{uuid.uuid4().hex[:8]}_{file.filename.replace(' ', '_')}"
        file_path = os.path.join(dated_upload_dir, unique_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_files_info.append({
            "original_name": file.filename,
            "saved_name": unique_filename,
            "saved_path": file_path,
            "file_size": os.path.getsize(file_path),
            "content_type": file.content_type
        })
    return saved_files_info

# ==========================================
# GENERAL ENDPOINTS
# ==========================================
@app.get("/", tags=["General"])
async def read_root():
    return {"success": True, "message": "Hospital AI Copilot API is running", "data": None}

@app.get("/health", tags=["General"])
async def health_check():
    client.admin.command("ping")
    return {"success": True, "message": "System healthy", "data": {"database": "connected"}}

# ==========================================
# REPORT UPLOAD & AI PIPELINE ENDPOINT
# ==========================================
@app.post("/reports/upload", tags=["Reports"], status_code=status.HTTP_201_CREATED)
async def upload_reports(
    files: List[UploadFile] = File(..., description="Select PDF files")
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Validate uploads
    for file in files:
        if file.content_type not in ALLOWED_FILE_TYPES or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not a valid PDF.")
        validate_file_size(file)

    create_upload_folder()
    saved_files_info = await save_uploaded_files(files)

    # ----------------- AI PIPELINE -----------------
    try:
        # Extract Text
        saved_paths = [info["saved_path"] for info in saved_files_info]
        pdf_texts_dict = process_uploaded_pdfs(saved_paths)
        combined_report_text = "\n\n".join(pdf_texts_dict.values())

        # Extract Patient Data
        patient_info, missing_fields = extract_patient_information(combined_report_text)

        # Deduplicate & Create Patient Document
        search_query = {}
        if patient_info.get("patient_name"):
            search_query["patient_name"] = {"$regex": f"^{patient_info['patient_name']}$", "$options": "i"}
        if patient_info.get("age") is not None:
            search_query["age"] = patient_info["age"]
        if patient_info.get("gender"):
            search_query["gender"] = {"$regex": f"^{patient_info['gender']}$", "$options": "i"}

        existing_patient = patients_collection.find_one(search_query) if search_query else None

        if existing_patient:
            patient_id = str(existing_patient["_id"])
            patient_data = existing_patient
        else:
            patient_data = {
                "patient_name": patient_info.get("patient_name") or "Unknown Patient",
                "age": patient_info.get("age") or 0,
                "gender": patient_info.get("gender") or "Unknown",
                "blood_group": patient_info.get("blood_group"),
                "disease": patient_info.get("disease"),
                "medical_history": patient_info.get("medical_history"),
                "created_at": datetime.utcnow()
            }
            patient_id = str(patients_collection.insert_one(patient_data).inserted_id)
            patient_data["_id"] = patient_id

        # Save Report Documents
        report_docs = []
        for info in saved_files_info:
            report_data = {
                "patient_id": patient_id,
                "file_name": info["original_name"],
                "file_path": info["saved_path"],
                "file_size": info["file_size"],
                "content_type": info["content_type"],
                "ai_processed": False,
                "upload_date": datetime.utcnow()
            }
            report_data["_id"] = str(reports_collection.insert_one(report_data).inserted_id)
            report_docs.append(report_data)

        # RAG, Visualizations & AI Analysis
        generate_visualizations(pdf_texts_dict)
        initialize_ai_rag()

        similar_cases, similarity_tips = get_similarity_context(
            combined_report_text, rag_embedder, rag_faiss_index, rag_historical_texts
        )
        try:
            doc1_analysis = doctor1_chatbot(
                "Provide a medical analysis.", combined_report_text, similar_cases, similarity_tips
            )
        except Exception as e:
            logger.warning(f"Doctor 1 AI analysis failed: {e}")
            doc1_analysis = "Doctor 1 analysis is temporarily unavailable."

        try:
            doc2_analysis = doctor2_chatbot(
                "Provide a medical analysis.", combined_report_text, similar_cases, similarity_tips
            )
        except Exception as e:
            logger.warning(f"Doctor 2 AI analysis failed: {e}")
            doc2_analysis = "Doctor 2 analysis is temporarily unavailable."

        # Mark Reports as Processed
        reports_collection.update_many(
            {"_id": {"$in": [ObjectId(r["_id"]) for r in report_docs]}},
            {"$set": {"ai_processed": True}}
        )
        for r in report_docs:
            r["ai_processed"] = True

        # Generate conversation ID for continuous chat
        conversation_id = str(uuid.uuid4())

        return {
            "success": True,
            "message": "AI Pipeline completed successfully",
            "data": {
                "conversation_id": conversation_id,
                "patient": serialize_doc(patient_data),
                "reports": report_docs,
                "missing_fields": missing_fields,
                "similar_cases": similar_cases,
                "doctor1": doc1_analysis,
                "doctor2": doc2_analysis
            }
        }
    except Exception as e:
        logger.error(f"AI Pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Files saved, but AI processing failed.")


# ==========================================
# CHAT ENDPOINT FOR CONTINUOUS CONSULTATIONS
# ==========================================
@app.post("/chat", tags=["Chat"], summary="Continue AI Consultation")
async def continue_ai_consultation(
    payload: dict = Body(...)
):
    """
    Allows doctors to continue asking questions about an already processed patient report.
    """
    patient_id = payload.get("patient_id")
    conversation_id = payload.get("conversation_id")
    doctor_role = payload.get("doctor_role", "Doctor 1")
    question = payload.get("question")

    # Step 1: Validate patient_id
    if not patient_id:
        raise HTTPException(status_code=400, detail="patient_id is required.")

    try:
        obj_id = ObjectId(patient_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid patient_id format.")

    # Step 2: Retrieve patient document from MongoDB
    patient = patients_collection.find_one({"_id": obj_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    # Step 3: Retrieve all report documents belonging to this patient
    reports = list(reports_collection.find({"patient_id": patient_id}))
    if not reports:
        raise HTTPException(status_code=404, detail="No reports found for this patient.")

    # Step 4: Reconstruct complete report text using process_uploaded_pdfs()
    report_paths = [r["file_path"] for r in reports if "file_path" in r]
    if not report_paths:
        raise HTTPException(status_code=404, detail="Report file paths are missing.")

    pdf_texts_dict = process_uploaded_pdfs(report_paths)
    report_text = "\n\n".join(pdf_texts_dict.values())

    # Generate or reuse conversation_id
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    # Step 5: Retrieve previous messages from chats_collection using conversation_id
    chat_docs = list(chats_collection.find({"conversation_id": conversation_id}).sort("created_at", 1))
    conversation_history = [
        {
            "role": doc.get("role"),
            "message": doc.get("message")
        }
        for doc in chat_docs
    ]

    # Step 6: Initialize RAG
    initialize_ai_rag()

    # Step 7: Generate similar_cases using get_similarity_context()
    similar_cases, _ = get_similarity_context(
        report_text,
        rag_embedder,
        rag_faiss_index,
        rag_historical_texts
    )

    # Prepare patient information dictionary for chat context
    patient_information = {
        "patient_name": patient.get("patient_name"),
        "age": patient.get("age"),
        "gender": patient.get("gender"),
        "blood_group": patient.get("blood_group"),
        "disease": patient.get("disease"),
        "medical_history": patient.get("medical_history")
    }

    # Step 8: Call chat_with_patient_context()
    try:
        ai_response = chat_with_patient_context(
            patient_information=patient_information,
            report_text=report_text,
            similar_cases=similar_cases,
            conversation_history=conversation_history,
            user_question=question,
            doctor_role=doctor_role
        )
    except Exception as e:
        logger.error(f"Chat generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate chat response.")

    now = datetime.utcnow()
    doctor_id = patient.get("doctor_id")

    # Step 9: Save TWO MongoDB documents (user message and assistant response) using Chat model
    user_chat_msg = Chat(
        patient_id=patient_id,
        doctor_id=doctor_id,
        conversation_id=conversation_id,
        role="user",
        message=question,
        created_at=now
    )
    assistant_chat_msg = Chat(
        patient_id=patient_id,
        doctor_id=doctor_id,
        conversation_id=conversation_id,
        role="assistant",
        message=ai_response,
        created_at=now
    )

    chats_collection.insert_one(user_chat_msg.model_dump())
    chats_collection.insert_one(assistant_chat_msg.model_dump())

    total_messages = chats_collection.count_documents({"conversation_id": conversation_id})

    # Step 10: Return formatted response
    return {
        "success": True,
        "message": "Chat response generated successfully.",
        "data": {
            "conversation_id": conversation_id,
            "answer": ai_response,
            "doctor_role": doctor_role,
            "timestamp": now.isoformat(),
            "message_count": total_messages
        }
    }


# ==========================================
# PATIENT CRUD ENDPOINTS
# ==========================================
@app.post("/patients", tags=["Patients"], status_code=status.HTTP_201_CREATED)
async def create_patient(patient: Patient):
    patient_id = str(patients_collection.insert_one(patient.model_dump()).inserted_id)
    return {
        "success": True,
        "message": "Patient created successfully",
        "data": {"patient_id": patient_id}
    }

@app.get("/patients", tags=["Patients"])
async def get_all_patients():
    patients = [serialize_doc(p) for p in patients_collection.find()]
    return {
        "success": True,
        "message": "Patients retrieved successfully",
        "data": patients
    }

@app.get("/patients/search", tags=["Patients"])
async def search_patients(
    patient_name: Optional[str] = Query(None),
    disease: Optional[str] = Query(None)
):
    query = {}
    if patient_name:
        query["patient_name"] = {"$regex": patient_name, "$options": "i"}
    if disease:
        query["disease"] = {"$regex": disease, "$options": "i"}

    patients = [serialize_doc(p) for p in patients_collection.find(query)]
    return {
        "success": True,
        "message": "Search completed",
        "data": patients
    }

@app.get("/patients/{patient_id}", tags=["Patients"])
async def get_patient(patient_id: str):
    try:
        obj_id = ObjectId(patient_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid ID format")

    patient = patients_collection.find_one({"_id": obj_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    return {
        "success": True,
        "message": "Patient retrieved",
        "data": serialize_doc(patient)
    }

@app.patch("/patients/{patient_id}", tags=["Patients"])
async def update_patient(patient_id: str, patient: PatientUpdate):
    try:
        obj_id = ObjectId(patient_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid ID format")

    update_data = patient.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields provided")

    result = patients_collection.update_one({"_id": obj_id}, {"$set": update_data})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Patient not found")

    return {
        "success": True,
        "message": "Patient updated successfully",
        "data": None
    }

@app.delete("/patients/{patient_id}", tags=["Patients"])
async def delete_patient(patient_id: str):
    try:
        obj_id = ObjectId(patient_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid ID format")

    result = patients_collection.delete_one({"_id": obj_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Patient not found")

    return {
        "success": True,
        "message": "Patient deleted successfully",
        "data": None
    }
