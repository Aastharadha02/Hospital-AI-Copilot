from ai import *

pdf_path = r"C:\Users\Aastha\Desktop\hospital\pdfs\Patient_1 (2).pdf"

print("=" * 60)
print("STEP 1: Extracting PDF Text")
print("=" * 60)

text = extract_text(pdf_path)

print(text[:1000])   # Print first 1000 characters

print("\n\n")

print("=" * 60)
print("STEP 2: Extracting Patient Information")
print("=" * 60)

patient_info, missing_fields = extract_patient_information(text)

print("\nExtracted Patient Information:\n")
print(patient_info)

print("\nMissing Fields:\n")
print(missing_fields)

print("\n\n")

print("=" * 60)
print("STEP 3: Creating Historical Database")
print("=" * 60)

historical_df = initialize_historical_database()

print(historical_df.head())

print("\n\n")

print("=" * 60)
print("STEP 4: Building FAISS Index")
print("=" * 60)

historical_texts = historical_df["Patient_Summary"].tolist()

embedder, faiss_index = build_faiss_index(historical_texts)

print("FAISS Index Created Successfully!")

print("\n\n")

print("=" * 60)
print("STEP 5: Finding Similar Patients")
print("=" * 60)

similar_cases, similarity_tips = get_similarity_context(
    text,
    embedder,
    faiss_index,
    historical_texts
)

print("\nSimilar Cases:\n")

for case in similar_cases:
    print(case)
    print("-" * 40)

print("\nSimilarity Tips:\n")
print(similarity_tips)

print("\n\n")

print("=" * 60)
print("STEP 6: Doctor 1")
print("=" * 60)

doctor1_response = doctor1_chatbot(
    "Give me a clinical summary.",
    text,
    similar_cases,
    similarity_tips
)

print(doctor1_response)

print("\n\n")

print("=" * 60)
print("STEP 7: Doctor 2")
print("=" * 60)

doctor2_response = doctor2_chatbot(
    "What should I pay attention to during handover?",
    text,
    similar_cases,
    similarity_tips
)

print(doctor2_response)

print("\n\n")

print("=" * 60)
print("AI PIPELINE TEST COMPLETED")
print("=" * 60)