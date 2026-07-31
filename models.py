from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

# ==========================================
# USER MODEL
# ==========================================
class User(BaseModel):
    """
    Model representing a hospital staff member or doctor.
    """
    name: str = Field(..., description="Full name of the user", examples=["Dr. Sarah Connor"])
    email: str = Field(..., description="Email address for login", examples=["sarah.connor@hospital.com"])
    hashed_password: str = Field(..., description="Secure, hashed password", examples=["$2b$12$EixZaYVK1fsbw1Zfln4V..."])
    role: str = Field(..., description="Assigned role (e.g., Doctor 1, Doctor 2)", examples=["Doctor 1"])
    created_at: datetime = Field(
        default_factory=datetime.utcnow, 
        description="Timestamp of user registration"
    )

# ==========================================
# PATIENT MODEL (FOR CREATION)
# ==========================================
class Patient(BaseModel):
    """
    Model representing a patient's core demographic and medical data.
    """
    doctor_id: Optional[str] = Field(
        default=None,
        description="MongoDB ObjectId of the assigned doctor",
        examples=["64c1b2f9e4b0a1a2b3c4d5e1"]
    )
    patient_name: str = Field(..., description="Full name of the patient", examples=["John Doe"])
    age: int = Field(..., description="Age of the patient in years", examples=[58])
    gender: str = Field(..., description="Biological sex/gender", examples=["Male"])
    
    blood_group: Optional[str] = Field(default=None, description="Blood type", examples=["O+"])
    disease: Optional[str] = Field(default=None, description="Primary diagnosis or active condition", examples=["Type 2 Diabetes Mellitus"])
    medical_history: Optional[str] = Field(
        default=None, 
        description="Text summary of past comorbidities and allergies", 
        examples=["Hypertension, Allergy to Penicillin"]
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow, 
        description="Timestamp of patient record creation"
    )

# ==========================================
# PATIENT UPDATE MODEL (FOR PARTIAL UPDATES)
# ==========================================
class PatientUpdate(BaseModel):
    """
    Model representing optional fields for performing partial updates on a patient.
    """
    doctor_id: Optional[str] = Field(
        default=None,
        description="MongoDB ObjectId of the assigned doctor",
        examples=["64c1b2f9e4b0a1a2b3c4d5e1"]
    )
    patient_name: Optional[str] = Field(
        default=None, 
        description="Full name of the patient", 
        examples=["John Doe"]
    )
    age: Optional[int] = Field(
        default=None, 
        description="Age of the patient in years", 
        examples=[58]
    )
    gender: Optional[str] = Field(
        default=None, 
        description="Biological sex/gender", 
        examples=["Male"]
    )
    blood_group: Optional[str] = Field(
        default=None, 
        description="Blood type", 
        examples=["O+"]
    )
    disease: Optional[str] = Field(
        default=None, 
        description="Primary diagnosis or active condition", 
        examples=["Type 2 Diabetes Mellitus"]
    )
    medical_history: Optional[str] = Field(
        default=None, 
        description="Text summary of past comorbidities and allergies", 
        examples=["Hypertension, Allergy to Penicillin"]
    )

# ==========================================
# REPORT MODEL
# ==========================================
class Report(BaseModel):
    """
    Model for tracking uploaded patient medical records (e.g., PDFs).
    """
    patient_id: str = Field(..., description="MongoDB ObjectId string referencing the Patient", examples=["64c1b2f9e4b0a1a2b3c4d5e6"])
    file_name: str = Field(..., description="Name of the uploaded PDF file", examples=["patient_john_doe_visit_3.pdf"])
    file_path: str = Field(..., description="Absolute or relative path where the uploaded PDF is stored", examples=["/uploads/reports/patient_john_doe_visit_3.pdf"])
    file_size: Optional[int] = Field(default=None, description="File size in bytes", examples=[1048576])
    content_type: Optional[str] = Field(default=None, description="MIME type of the uploaded file (for example application/pdf)", examples=["application/pdf"])
    ai_processed: bool = Field(default=False, description="Indicates whether the uploaded report has been processed by the AI pipeline")
    upload_date: datetime = Field(
        default_factory=datetime.utcnow, 
        description="Timestamp of when the report was uploaded"
    )

# ==========================================
# CHAT MODEL
# ==========================================
class Chat(BaseModel):
    """
    Model for storing individual messages within continuous AI chat consultations.
    """
    patient_id: str = Field(..., description="MongoDB ObjectId of the patient", examples=["64c1b2f9e4b0a1a2b3c4d5e6"])
    doctor_id: Optional[str] = Field(
        default=None,
        description="MongoDB ObjectId of the logged-in doctor",
        examples=["64c1b2f9e4b0a1a2b3c4d5e1"]
    )
    conversation_id: str = Field(..., description="A UUID string grouping messages belonging to one consultation", examples=["123e4567-e89b-12d3-a456-426614174000"])
    role: str = Field(..., description="Indicates who sent the message ('user' or 'assistant')", examples=["user"])
    message: str = Field(..., description="Complete message text", examples=["Are there any known drug interactions for the prescribed dosage?"])
    created_at: datetime = Field(
        default_factory=datetime.utcnow, 
        description="Automatically generated timestamp for the message"
    )