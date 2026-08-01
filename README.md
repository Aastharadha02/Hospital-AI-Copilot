#  Hospital AI Copilot

An AI-powered full-stack web application that assists healthcare professionals in analyzing patient medical reports. The system extracts patient information from PDF reports, retrieves similar historical cases using Retrieval-Augmented Generation (RAG), and generates AI-assisted clinical insights through Large Language Models (LLMs).

---

##  Features

- Upload one or multiple patient PDF reports
- Automatic PDF text extraction
- AI-powered patient information extraction
- Similar historical case retrieval using FAISS
- Retrieval-Augmented Generation (RAG)
- AI-assisted medical analysis
- Continuous doctor-AI chat
- Patient record management (CRUD)
- MongoDB integration
- RESTful API using FastAPI
- Responsive React frontend

---

##  Tech Stack

### Frontend
- React.js
- Axios
- HTML5
- CSS3

### Backend
- FastAPI
- Python

### Database
- MongoDB

### AI & Machine Learning
- Groq LLM
- FAISS
- Sentence Transformers
- Retrieval-Augmented Generation (RAG)

### Libraries
- PyMuPDF
- Pydantic
- Matplotlib
- Uvicorn

---

##  Project Structure

```
Hospital-AI-Copilot
│
├── backend
│   ├── main.py
│   ├── ai.py
│   ├── database.py
│   ├── config.py
│   ├── models.py
│   ├── requirements.txt
│   └── uploads/
│
├── frontend
│   ├── src
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── README.md
└── .gitignore
```

---

##  Installation

### Backend

```bash
cd backend

python -m venv venv

# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI
uvicorn main:app --reload
```

Backend URL:

```
http://127.0.0.1:8000
```

Swagger API Documentation:

```
http://127.0.0.1:8000/docs
```

---

### Frontend

```bash
cd frontend

npm install

npm run dev
```

Frontend URL:

```
http://localhost:5173
```

---

##  Workflow

```
Upload PDF Reports
        │
        ▼
Extract Report Text
        │
        ▼
AI Patient Information Extraction
        │
        ▼
Store Patient Data in MongoDB
        │
        ▼
Retrieve Similar Historical Cases (FAISS + RAG)
        │
        ▼
Generate AI Medical Analysis
        │
        ▼
Continuous Doctor-AI Chat
```

---

##  API Endpoints

### General
- `GET /`
- `GET /health`

### Reports
- `POST /reports/upload`

### Patients
- `POST /patients`
- `GET /patients`
- `GET /patients/{id}`
- `PATCH /patients/{id}`
- `DELETE /patients/{id}`
- `GET /patients/search`

### AI Chat
- `POST /chat/doctor1`
- `POST /chat/doctor2`

---

##  Future Enhancements

- User authentication (JWT)
- Doctor dashboard
- Cloud deployment
- Docker support
- Electronic Health Record (EHR) integration
- Drug interaction analysis
- Multi-language support

---

##  Author

**Aastha Hotwani**

B.Tech in Information and Communication Technology (ICT)

Pandit Deendayal Energy University (PDEU)

---

## 📄 License

This project is developed for educational and research purposes only.
