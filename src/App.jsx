import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  // New states for chat and consultation
  const [conversationId, setConversationId] = useState(null);
  const [chatMessages, setChatMessages] = useState([]);
  const [currentQuestion, setCurrentQuestion] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  const handleFileChange = (event) => {
    if (event.target.files) {
      // Convert FileList object to an array
      setSelectedFiles(Array.from(event.target.files));
    }
  };

  const handleUpload = async () => {
    // Prevent upload if no files are selected
    if (selectedFiles.length === 0) return;

    setLoading(true);

    // Create FormData and append each selected file
    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append("files", file);
    });

    try {
      // Send the request to the FastAPI backend
      const response = await axios.post("http://127.0.0.1:8000/reports/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      console.log("SUCCESS RESPONSE:", response);
      console.log("STATUS:", response.status);
      console.log("DATA:", response.data);
      console.log("RESULT:", response.data.data);

      // Store response data, conversation_id, and log to console
      setResult(response.data.data);
      setConversationId(response.data.data.conversation_id);
    } catch (error) {
      console.error("UPLOAD ERROR:", error);
      console.error("ERROR MESSAGE:", error.message);
      console.error("ERROR RESPONSE:", error.response);
      console.error("ERROR STATUS:", error.response?.status);
      console.error("ERROR DATA:", error.response?.data);
      if (error.response?.data?.detail) {
        console.error("ERROR DETAIL:", error.response?.data?.detail);
      }

      alert(error.response?.data?.detail || error.message);
    } finally {
      // Always stop the loading spinner
      setLoading(false);
    }
  };

  const handleSendChat = async () => {
    if (!currentQuestion.trim() || !result || !result.patient) return;

    setChatLoading(true);
    const questionToSend = currentQuestion;

    try {
      const response = await axios.post("http://127.0.0.1:8000/chat", {
        patient_id: result.patient._id,
        conversation_id: conversationId,
        doctor_role: "Doctor 1",
        question: questionToSend
      });

      const aiAnswer = response.data.data.answer;

      // Append user question and AI answer into chatMessages
      setChatMessages((prevMessages) => [
        ...prevMessages,
        { role: "user", message: questionToSend },
        { role: "assistant", message: aiAnswer }
      ]);

      // Clear textarea
      setCurrentQuestion("");
    } catch (error) {
      alert(error.response?.data?.detail || "Chat request failed");
    } finally {
      setChatLoading(false);
    }
  };

  const handleEndConsultation = () => {
    setResult(null);
    setConversationId(null);
    setChatMessages([]);
    setCurrentQuestion("");
    setSelectedFiles([]);
  };

  return (
    <div className="app-container">
      {/* Header Section */}
      <header className="header">
        <h1>Hospital AI Copilot</h1>
        <p>Upload one or more patient PDF reports for AI analysis.</p>
      </header>

      {/* Upload Section */}
      <section className="card upload-section">
        <div className="upload-controls">
          <label className="file-input-label">
            Choose Files
            <input 
              type="file" 
              multiple 
              accept=".pdf" 
              onChange={handleFileChange} 
              className="hidden-file-input"
            />
          </label>
          
          <button 
            className="btn-upload" 
            onClick={handleUpload} 
            disabled={selectedFiles.length === 0 || loading}
          >
            Upload
          </button>
        </div>

        {selectedFiles.length > 0 && (
          <div className="selected-files">
            <h3>Selected Files:</h3>
            <ul>
              {selectedFiles.map((file, index) => (
                <li key={index}>- {file.name}</li>
              ))}
            </ul>
          </div>
        )}
      </section>

      {/* Loading Section */}
      {loading && (
        <section className="loading-section card">
          <div className="spinner"></div>
          <p>Processing reports...</p>
        </section>
      )}

      {/* Patient Information Card */}
      <section className="card">
        <h2>Patient Information</h2>
        {result && result.patient ? (
          <div className="patient-info-grid">
            <p><strong>Name:</strong> {result.patient.patient_name || "Not Available"}</p>
            <p><strong>Age:</strong> {result.patient.age !== null && result.patient.age !== undefined ? result.patient.age : "Not Available"}</p>
            <p><strong>Gender:</strong> {result.patient.gender || "Not Available"}</p>
            <p><strong>Disease:</strong> {result.patient.disease || "Not Available"}</p>
            <p><strong>Blood Group:</strong> {result.patient.blood_group || "Not Available"}</p>
            <p><strong>Medical History:</strong> {result.patient.medical_history || "Not Available"}</p>
          </div>
        ) : (
          <p className="placeholder-text">No patient data available.</p>
        )}
      </section>

      {/* Missing Fields Card */}
      <section className="card">
        <h2>Missing Information</h2>
        {result && result.missing_fields && result.missing_fields.length > 0 ? (
          <ul>
            {result.missing_fields.map((field, index) => (
              <li key={index}>{field}</li>
            ))}
          </ul>
        ) : (
          <p className="placeholder-text">No missing fields.</p>
        )}
      </section>

      {/* Similar Cases Card */}
      <section className="card">
        <h2>Similar Historical Cases</h2>
        {result && result.similar_cases && result.similar_cases.length > 0 ? (
          <div>
            {result.similar_cases.map((caseItem, index) => (
              <div key={index} style={{ border: '1px solid #ccc', padding: '10px', marginBottom: '10px', borderRadius: '6px' }}>
                <p style={{ margin: 0 }}>{caseItem}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="placeholder-text">No similar historical cases found.</p>
        )}
      </section>

      {/* Doctor 1 Card */}
      <section className="card">
        <h2>Doctor 1 Analysis</h2>
        {result && result.doctor1 ? (
          <div style={{ maxHeight: '300px', overflowY: 'auto', whiteSpace: 'pre-wrap' }}>
            {result.doctor1}
          </div>
        ) : (
          <p className="placeholder-text">No analysis available.</p>
        )}
      </section>

      {/* Doctor 2 Card */}
      <section className="card">
        <h2>Doctor 2 Analysis</h2>
        {result && result.doctor2 ? (
          <div style={{ maxHeight: '300px', overflowY: 'auto', whiteSpace: 'pre-wrap' }}>
            {result.doctor2}
          </div>
        ) : (
          <p className="placeholder-text">No analysis available.</p>
        )}
      </section>

      {/* Uploaded Reports Card */}
      <section className="card">
        <h2>Uploaded Reports</h2>
        {result && result.reports && result.reports.length > 0 ? (
          <ul>
            {result.reports.map((report, index) => (
              <li key={index} style={{ marginBottom: '10px' }}>
                <strong>File Name:</strong> {report.file_name} |&nbsp;
                <strong>Size:</strong> {report.file_size} bytes |&nbsp;
                <strong>AI Processed:</strong> {report.ai_processed ? "Yes" : "No"}
              </li>
            ))}
          </ul>
        ) : (
          <p className="placeholder-text">No uploaded reports available.</p>
        )}
      </section>

      {/* Continue Consultation Card */}
      {result && (
        <section className="card">
          <h2>Continue Consultation</h2>
          
          <div style={{ maxHeight: '350px', overflowY: 'auto', marginBottom: '15px', padding: '10px', backgroundColor: '#fdfdfd', border: '1px solid #eee', borderRadius: '6px' }}>
            {chatMessages.length === 0 ? (
              <p className="placeholder-text">No conversation messages yet. Ask a question below.</p>
            ) : (
              chatMessages.map((msg, index) => (
                <div key={index} style={{ marginBottom: '12px' }}>
                  <strong>{msg.role === 'user' ? 'Doctor:' : 'AI:'}</strong>
                  <div style={{ whiteSpace: 'pre-wrap', marginTop: '4px', paddingLeft: '10px' }}>
                    {msg.message}
                  </div>
                </div>
              ))
            )}
          </div>

          <textarea
            rows="3"
            placeholder="Ask anything about this patient..."
            value={currentQuestion}
            onChange={(e) => setCurrentQuestion(e.target.value)}
            disabled={chatLoading}
            style={{ width: '100%', padding: '10px', borderRadius: '6px', border: '1px solid #ccc', boxSizing: 'border-box', marginBottom: '10px', fontFamily: 'inherit' }}
          />

          <div style={{ display: 'flex', gap: '10px', justifyContent: 'space-between', alignItems: 'center' }}>
            <button
              className="file-input-label"
              onClick={handleSendChat}
              disabled={chatLoading || !currentQuestion.trim()}
              style={{ border: 'none' }}
            >
              {chatLoading ? "Thinking..." : "Send"}
            </button>

            <button
              onClick={handleEndConsultation}
              style={{ backgroundColor: '#dc3545', color: 'white', padding: '10px 20px', borderRadius: '6px', fontWeight: '500', cursor: 'pointer', border: 'none' }}
            >
              End Consultation
            </button>
          </div>
        </section>
      )}
    </div>
  );
}

export default App;