# Query Your Model

This project implements an end-to-end framework for interacting with machine learning models through natural-language–style queries. It combines FastAPI (for backend explainability services), Streamlit (for a conversational UI), and explainability methods like SHAP with retrieval-augmented reasoning to make predictions more transparent and interpretable. The system allows you to upload a trained model, input feature values for a new instance, generate explanations using SHAP feature attributions, retrieve similar past reasoning cases for context, and interact with the model via a simple chat-style interface.

##  Features
- FastAPI backend (`/explain` endpoint): loads a scikit-learn model, generates predictions and probabilities, computes SHAP explanations for the input instance, and retrieves top-K similar cases using cosine similarity over SHAP + feature vectors.  
- Streamlit UI: sidebar for uploading models, providing feature names, and configuring retrieval; main panel for conversational “chat” with the model; displays predictions, SHAP feature importance, bar plots, and similar cases.  
- Retrieval-Augmented Explainability: cases are stored in local indices (`features.npy`, `shap.npy`, `meta.jsonl`) and top-K reasoning cases are retrieved using weighted cosine similarity.  

##  Repository Layout
Query_Your_Model/  
├── app/  
│   ├── api_fastapi.py — FastAPI backend: /explain endpoint  
│   └── app_chat.py — Streamlit chat-like UI  
├── core/  
│   ├── model_loader.py — load_model(), predict()  
│   ├── explain.py — explain_instance(): SHAP explainer & top-k attribution logic  
│   ├── retrieval.py — combined_similarity, retrieve_topk, ood_score  
│   ├── storage.py — persistence helpers: features.npy, shap.npy, meta.jsonl  
│   ├── utils.py — helpers (safe probability conversion, etc.)  
│   └── schemas.py — Pydantic models for API (ExplainRequest, ExplainResponse, etc.)  
├── data/  
│   ├── base_indices/ — Prebuilt retrieval indices (features, shap, meta, index.jsonl)  
│   └── model_data/ — Example trained model (model.pkl, iris dataset)  
├── scripts/  
│   ├── build_base_index.py — Build retrieval indices from dataset + model  
│   ├── add_user_model.py — Register additional models  
│   ├── demo_predict.py — Simple CLI prediction & explanation  
│   └── build_iris.bat — Windows helper for building Iris index  
├── tests/  
│   └── test_similarity.py — Unit tests for combined_similarity  
└── README.md — Project documentation  

##  Setup & Installation
1. Clone this repo and create a virtual environment.  
2. Install dependencies from `requirements.txt`.  
3. Core dependencies include: fastapi, uvicorn, streamlit, scikit-learn, shap, pandas, numpy, matplotlib.  

##  Running the System
Start the FastAPI backend with `uvicorn Query_Your_Model.app.api_fastapi:app --reload`.  
API runs at `http://127.0.0.1:8000` and the `/explain` endpoint can be tested from the Swagger UI at `/docs`.  

Launch the Streamlit UI with `streamlit run app/ui_streamlit.py`.  
The UI runs at `http://localhost:8501`.  

