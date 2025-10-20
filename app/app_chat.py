import streamlit as st
import requests
import json

st.set_page_config(page_title="Chat with Your Model", page_icon="💬", layout="wide")
st.title("💬 Chat with Your Model")

# Sidebar for API + settings
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("FastAPI endpoint", value="http://127.0.0.1:8000/explain")
    model_path = st.text_input("Model path", value="Query_Your_Model/model_data/model.pkl")
    feat_names_str = st.text_input("Feature names (comma-separated)", 
                                   value="sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)")
    namespace = st.text_input("Namespace", value="Query_Your_Model/data/base_indices/iris_global")
    alpha = st.slider("Alpha", 0.0, 1.0, 0.7, 0.05)
    k = st.slider("Top-K similar", 1, 10, 5)

feat_names = [s.strip() for s in feat_names_str.split(",")]

# A helper to generate natural language
def explain_in_words(res):
    pred = res["prediction"]["y_pred"]  # already string now
    proba = res["prediction"]["proba"]
    exp = res["explanation"]
    topk = exp["topk"]

    msg = f"The model predicted **{pred}** with probability {proba:.2f}.\n\n"
    msg += "Key reasons:\n"
    for f in topk:
        effect = "increased" if f["shap"] > 0 else "decreased"
        msg += f"- {f['feature']} = {f['value']} which {effect} the prediction (impact {f['abs_impact']:.2f}).\n"

    if res.get("similar_cases"):
        msg += f"\nIt also found {len(res['similar_cases'])} similar past cases. Example:\n"
        case = res["similar_cases"][0]
        msg += f"- Case {case['case_id']} with features {case['features']} predicted as {case['y_pred']}."

    return msg


# Conversation memory
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display existing messages
for role, content in st.session_state["messages"]:
    with st.chat_message(role):
        st.markdown(content)

# Chat input
if user_q := st.chat_input("Ask your model something..."):
    st.session_state["messages"].append(("user", user_q))
    with st.chat_message("user"):
        st.markdown(user_q)

    # For demo, assume any user question → trigger /explain with dummy features
    payload = {
        "model_path": model_path,
        "feature_names": feat_names,
        "features": [5.1, 3.5, 1.4, 0.2],  # 👈 you can make this dynamic
        "namespace": namespace,
        "retrieval": {"alpha": alpha, "k": k, "use_retrieval": True, "namespace": namespace}
    }
    try:
        res = requests.post(api_url, json=payload).json()
        answer = explain_in_words(res)
    except Exception as e:
        answer = f"⚠️ Error contacting API: {e}"

    st.session_state["messages"].append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)
