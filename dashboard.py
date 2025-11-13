"""
Simple Streamlit dashboard for the Member Data QA Service
"""
import streamlit as st
import requests
import json

# Configuration
# Use local API for testing, or set to deployed URL for production
API_URL = "http://localhost:8000"  # Change to deployed URL if needed
# API_URL = "https://aurora-applied-ai-ml-engineer-take-home.onrender.com"

# Page config
st.set_page_config(
    page_title="Member Data QA System",
    page_icon=None,
    layout="centered"
)

# Title
st.title("Member Data Question-Answering System")
st.markdown("Ask questions about member data from the API")

# Example questions
example_questions = [
    "When is Layla planning her trip to London?",
    "How many cars does Vikram Desai have?",
    "What are Amira's favorite restaurants?",
    "What is Sophia planning?",
    "When is Fatima going to Paris?"
]

# Sidebar with examples
with st.sidebar:
    st.header("Example Questions")
    for example in example_questions:
        if st.button(example, key=example, use_container_width=True):
            st.session_state.question = example

# Main input
question = st.text_input(
    "Enter your question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g., When is Layla planning her trip to London?"
)

# Ask button
if st.button("Ask Question", type="primary", use_container_width=True):
    if question.strip():
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                answer = data.get("answer", "No answer received")
                confidence = data.get("confidence", 0.0)
                
                st.success("Answer:")
                st.info(answer)
                
                # Display confidence score with visual indicator
                st.markdown("**Confidence:**")
                confidence_label = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
                st.progress(confidence)
                st.caption(f"{confidence_label}: {confidence:.1%} confidence")
                
                # Store in session for history
                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.insert(0, {
                    "question": question, 
                    "answer": answer,
                    "confidence": confidence
                })
                
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {str(e)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a question")

# Show history
if "history" in st.session_state and st.session_state.history:
    st.divider()
    st.subheader("Recent Questions")
    for i, item in enumerate(st.session_state.history[:5]):  # Show last 5
        confidence = item.get("confidence", 0.0)
        confidence_label = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
        with st.expander(f"Q: {item['question']} [{confidence_label}: {confidence:.0%}]"):
            st.write(f"**Answer:** {item['answer']}")
            if "confidence" in item:
                st.caption(f"Confidence: {item['confidence']:.1%}")

# Footer
st.divider()
st.markdown("**API Endpoint:** " + API_URL)
try:
    health_check = requests.get(f"{API_URL}/health", timeout=5)
    status = "Online" if health_check.status_code == 200 else "Offline"
except:
    status = "Offline"
st.markdown("**Status:** " + status)

