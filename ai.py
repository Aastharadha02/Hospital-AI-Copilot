"""
Hospital AI Copilot - AI Pipeline Module
Handles text extraction, JSON formatting, visualization generation, and LLM orchestration.
"""

import os
import re
import json
import random
from typing import Dict, Tuple, List, Any

import PyPDF2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Force headless mode for backend processing
import matplotlib.pyplot as plt
import seaborn as sns

from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss

from config import logger, GROQ_API_KEY, AI_MODEL_NAME, FAISS_TOP_K, EMBEDDING_MODEL_NAME

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY is missing. AI functionality will fail.")
    raise ValueError("GROQ_API_KEY is not set.")

groq_client = Groq(api_key=GROQ_API_KEY)

# ==========================================
# PDF EXTRACTION FUNCTIONS
# ==========================================
def extract_text(pdf_path: str) -> str:
    """
    Extracts text from a single PDF file.
    
    Args:
        pdf_path (str): The OS path to the PDF file.
    Returns:
        str: The fully extracted string content.
    """
    reader = PyPDF2.PdfReader(pdf_path)
    text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
    return text

def process_uploaded_pdfs(file_paths: List[str]) -> Dict[str, str]:
    """
    Processes a list of PDF file paths.
    
    Args:
        file_paths (List[str]): List of absolute or relative file paths.
    Returns:
        Dict[str, str]: A dictionary mapping the base filenames to their extracted text.
    """
    return {os.path.basename(fname): extract_text(fname) for fname in file_paths}

# ==========================================
# INFORMATION EXTRACTION
# ==========================================
def extract_patient_information(report_text: str) -> Tuple[Dict[str, Any], List[str]]:
    """
    Extracts structured patient demographic and clinical information from raw report text.
    
    Args:
        report_text (str): The raw text extracted from PDFs.
    Returns:
        Tuple containing a JSON dictionary of extracted data and a list of missing fields.
    """
    prompt = f"""
You are a medical data extraction assistant.
Extract the patient information from the medical report text provided below.

REQUIREMENTS:
- Do not invent information.
- Do not guess.
- Extract only what is explicitly present.
- Return valid JSON only.
- No markdown formatting.
- No explanations.

If a field is not present, you MUST return null.

REQUIRED JSON STRUCTURE:
{{
    "patient_name": "",
    "age": null,
    "gender": "",
    "blood_group": "",
    "disease": "",
    "medical_history": "",
    "chief_complaint": "",
    "diagnosis": "",
    "medications": "",
    "allergies": ""
}}

REPORT TEXT:
{report_text}
"""
    fallback_data = {
        "patient_name": None, "age": None, "gender": None, "blood_group": None,
        "disease": None, "medical_history": None, "chief_complaint": None,
        "diagnosis": None, "medications": None, "allergies": None
    }

    try:
        response = groq_client.chat.completions.create(
            model=AI_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a strict data extraction AI. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content.strip()
        
        # Fallback to strip markdown if the LLM disobeys instructions
        if content.startswith("```json"): content = content[7:]
        if content.startswith("```"): content = content[3:]
        if content.endswith("```"): content = content[:-3]

        extracted_data = json.loads(content.strip())
        missing_fields = [key for key, value in extracted_data.items() if value is None]
        
        return extracted_data, missing_fields
        
    except Exception as e:
        logger.error(f"Patient information extraction failed: {e}", exc_info=True)
        return fallback_data, list(fallback_data.keys())

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================
def generate_visualizations(pdf_texts: Dict[str, str]) -> None:
    """
    Generates and saves visualizations for patient stability and consistency silently.
    
    Args:
        pdf_texts (Dict[str, str]): Dictionary mapping filenames to report text.
    """
    for fname, text in pdf_texts.items():
        visit_blocks = re.split(r"Visit\s*\d+", text, flags=re.IGNORECASE)[1:]
        visit_labels = [f"Visit {i+1}" for i in range(len(visit_blocks))]

        if not visit_blocks:
            continue

        stability_scores = [int(s) for s in re.findall(r"stability\s*score\s*:\s*(\d+)", text, flags=re.IGNORECASE)]
        stability_scores += [None] * max(0, len(visit_labels) - len(stability_scores))

        medical_keys = {"Diabetes": "diabetes", "Hypertension": "hypertension", "Chronic Kidney Disease": "kidney", "Drug Allergy": "allergy"}
        consistency_data = {visit_labels[i]: [int(key in visit.lower()) for key in medical_keys.values()] for i, visit in enumerate(visit_blocks)}
        consistency_df = pd.DataFrame(consistency_data, index=medical_keys.keys())

        risk_burden = []
        for i, visit in enumerate(visit_blocks):
            visit_lower = visit.lower()
            score = sum(1 for kw in ["diabetes", "hypertension", "kidney", "allergy"] if kw in visit_lower)
            if "smoker" in visit_lower or "smoking" in visit_lower: score += 1
            if i > 0 and stability_scores[i] is not None and stability_scores[i-1] is not None and stability_scores[i] < stability_scores[i-1]:
                score += 1
            risk_burden.append(score)

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        fig.suptitle(f"Patient Safety Overview — {fname}", fontsize=14)

        axes[0].plot(visit_labels, stability_scores, marker="o", linewidth=2)
        axes[0].set(ylim=(0, 10), title="Patient Stability Trend", ylabel="Stability Score")
        axes[0].grid(True, linestyle="--", alpha=0.6)

        sns.heatmap(consistency_df, annot=True, cmap="RdYlGn", cbar=False, linewidths=0.5, ax=axes[1])
        axes[1].set_title("Medical History Documentation Consistency")

        axes[2].bar(visit_labels, risk_burden, color="orange", edgecolor="black")
        axes[2].set(title="Clinical Risk Burden per Visit", ylabel="Number of Risk Factors", ylim=(0, max(risk_burden) + 1 if risk_burden else 5))

        plt.tight_layout()
        # Matplotlib 'Agg' prevents blocking. Plots are processed in memory.
        plt.close(fig) 

# ==========================================
# HISTORICAL DATABASE & RAG INDEX
# ==========================================
def generate_historical_database(num_patients: int = 100) -> pd.DataFrame:
    """Generates a synthetic historical patient database DataFrame."""
    random.seed(42)
    primary_conditions = ["Type 2 Diabetes Mellitus", "Hypertension", "Chronic Kidney Disease", "Ischemic Heart Disease", "Sepsis", "Pneumonia"]
    comorbidities = ["Hypertension", "Chronic Kidney Disease", "Obesity", "Asthma", "Ischemic Heart Disease", "None"]
    allergies = [("Sulfa drugs", "severe rash"), ("Penicillin", "anaphylaxis"), ("NSAIDs", "gastric bleed"), ("None", None)]
    lessons = ["Proper allergy documentation prevented adverse drug reactions.", "Incomplete handover led to delayed recognition of allergy.", "Early specialist consultation improved outcomes.", "Medication reconciliation reduced complications.", "Poor documentation increased patient safety risk."]

    historical_patients = []
    for i in range(1, num_patients + 1):
        age, gender = random.randint(45, 80), random.choice(["male", "female"])
        primary, comorb = random.choice(primary_conditions), random.choice(comorbidities)
        allergy, reaction = random.choice(allergies)
        outcome, lesson = random.randint(3, 9), random.choice(lessons)

        summary = f"{age}-year-old {gender} admitted with {primary}. Comorbidities included {comorb}. "
        summary += f"Documented allergy to {allergy} causing {reaction}. " if allergy != "None" else "No known drug allergies at admission. "
        summary += f"Treatment decisions varied based on handover quality. Outcome score {outcome}/10. Key lesson: {lesson}"

        historical_patients.append({"Patient_ID": f"HIST_{i:03d}", "Patient_Summary": summary.strip()})

    return pd.DataFrame(historical_patients)

def build_faiss_index(historical_texts: List[str]) -> Tuple[SentenceTransformer, faiss.IndexFlatL2]:
    """Creates embeddings and builds a FAISS index from historical patient texts."""
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedder.encode(historical_texts, show_progress_bar=False)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(np.array(embeddings))
    return embedder, faiss_index

# ==========================================
# SIMILARITY RETRIEVAL
# ==========================================
def get_similarity_context(current_patient_text: str, embedder: SentenceTransformer, faiss_index: faiss.IndexFlatL2, historical_texts: List[str]) -> Tuple[List[str], str]:
    """Retrieves similar cases using FAISS and generates safety tips using Groq."""
    query_embedding = embedder.encode([current_patient_text])
    _, indices = faiss_index.search(query_embedding, FAISS_TOP_K)
    similar_cases = [historical_texts[i] for i in indices[0]]

    tips_prompt = f"""
You are a clinical decision-support assistant.
CURRENT PATIENT SUMMARY:
{current_patient_text}

SIMILAR HISTORICAL PATIENT CASES:
{chr(10).join(similar_cases)}

Task:
- Identify what worked well
- Identify what went wrong
- Generate concise safety-focused tips
"""
    response = groq_client.chat.completions.create(
        model=AI_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a medical safety assistant."},
            {"role": "user", "content": tips_prompt}
        ]
    )
    return similar_cases, response.choices[0].message.content

# ==========================================
# DOCTOR CHATBOTS
# ==========================================
def doctor1_chatbot(question: str, current_patient_corpus: str, similar_cases: List[str], similarity_tips: str) -> str:
    """AI Copilot analysis acting as the Treating Doctor."""
    prompt = f"""
You are Doctor 1's AI Copilot.
ROLE: Treating doctor during active shift
CURRENT PATIENT DATA: {current_patient_corpus}
SIMILAR PAST PATIENTS: {chr(10).join(similar_cases)}
LESSONS FROM SIMILAR PATIENTS: {similarity_tips}
QUESTION: {question}
Rules: No diagnosis. Clearly reference similar past patients when relevant.
"""
    response = groq_client.chat.completions.create(
        model=AI_MODEL_NAME,
        messages=[{"role": "system", "content": "You assist the treating doctor."}, {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def doctor2_chatbot(question: str, current_patient_corpus: str, similar_cases: List[str], similarity_tips: str) -> str:
    """AI Copilot analysis acting as the Incoming / Handover Doctor."""
    prompt = f"""
You are Doctor 2's AI Copilot.
ROLE: Incoming doctor during handover
CURRENT PATIENT DATA: {current_patient_corpus}
SIMILAR PAST PATIENTS: {chr(10).join(similar_cases)}
LESSONS FROM SIMILAR PATIENTS: {similarity_tips}
QUESTION: {question}
Rules: No diagnosis. Highlight risks seen in similar past patients.
"""
    response = groq_client.chat.completions.create(
        model=AI_MODEL_NAME,
        messages=[{"role": "system", "content": "You assist the incoming doctor."}, {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ==========================================
# CONTINUOUS CHAT CONTEXT
# ==========================================
def chat_with_patient_context(
    patient_information: dict,
    report_text: str,
    similar_cases: list,
    conversation_history: list,
    user_question: str,
    doctor_role: str = "Doctor 1"
) -> str:
    """
    Generates a follow-up response for continuous AI conversations based on patient context,
    RAG cases, previous conversation history, and role-specific instructions.
    
    Args:
        patient_information (dict): Structured patient demographics and clinical details.
        report_text (str): Complete extracted report text.
        similar_cases (list): List of similar past patient summaries from FAISS/RAG.
        conversation_history (list): List of preceding message dictionaries (e.g., role and message).
        user_question (str): The latest query asked by the doctor.
        doctor_role (str): Specifies whether to act as "Doctor 1" or "Doctor 2".
    
    Returns:
        str: The AI assistant's response text.
    """
    # Validate doctor_role, default to "Doctor 1" if invalid
    if doctor_role not in ["Doctor 1", "Doctor 2"]:
        doctor_role = "Doctor 1"

    # Format conversation history into a readable transcript string
    formatted_history = ""
    if conversation_history:
        history_lines = []
        for msg in conversation_history:
            role = msg.get("role", "user")
            text = msg.get("message", "")
            prefix = "Doctor" if role == "user" else "Assistant"
            history_lines.append(f"{prefix}: {text}")
        formatted_history = "\n".join(history_lines)
    else:
        formatted_history = "No previous conversation."

    # Format patient information dictionary into a summary string
    patient_summary_lines = [f"{k}: {v}" for k, v in patient_information.items()]
    patient_summary_str = "\n".join(patient_summary_lines)

    # Handle similar_cases display requirement if empty
    if not similar_cases:
        similar_cases_str = "No similar historical cases found."
    else:
        similar_cases_str = chr(10).join(similar_cases)

    # Role-specific instructions and prompt styling matching doctor1_chatbot and doctor2_chatbot
    if doctor_role == "Doctor 2":
        role_description = "Incoming doctor during handover"
        copilot_title = "Doctor 2's AI Copilot"
        rule_instruction = (
            "- No diagnosis\n"
            "- Highlight risks seen in similar past patients\n"
            "- Answer only using the available patient information, report, conversation history, and similar historical cases.\n"
            "- If information is unavailable, clearly state that.\n"
            "- Do not invent medical facts.\n"
            "- Do not fabricate laboratory values or medications.\n"
            "- Do not make a final diagnosis.\n"
            "- Keep responses concise and clinically relevant."
        )
        system_content = "You assist the incoming doctor."
    else:
        role_description = "Treating doctor during active shift"
        copilot_title = "Doctor 1's AI Copilot"
        rule_instruction = (
            "- No diagnosis\n"
            "- Clearly reference similar past patients when relevant\n"
            "- Answer only using the available patient information, report, conversation history, and similar historical cases.\n"
            "- If information is unavailable, clearly state that.\n"
            "- Do not invent medical facts.\n"
            "- Do not fabricate laboratory values or medications.\n"
            "- Do not make a final diagnosis.\n"
            "- Keep responses concise and clinically relevant."
        )
        system_content = "You assist the treating doctor."

    prompt = f"""
You are {copilot_title}.

ROLE:
- {role_description}

PATIENT INFORMATION:
{patient_summary_str}

CURRENT PATIENT DATA (ORIGINAL REPORT):
{report_text}

SIMILAR PAST PATIENTS (FROM HISTORY DATABASE):
{similar_cases_str}

CONVERSATION HISTORY:
{formatted_history}

LATEST QUESTION:
{user_question}

Rules:
{rule_instruction}
"""

    response = groq_client.chat.completions.create(
        model=AI_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content