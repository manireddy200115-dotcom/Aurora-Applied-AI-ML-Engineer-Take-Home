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
        # Show progress message
        progress_placeholder = st.empty()
        progress_placeholder.info("🔄 Processing question... Loading relevant messages and computing embeddings (this may take 60-120 seconds for first request)")
        
        try:
            # Increased timeout for on-demand message loading and embedding computation
            # First request may take longer (120s) due to model loading and data fetching
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question},
                timeout=120  # 120 seconds for first request (model loading, message fetching, embedding computation)
            )
            progress_placeholder.empty()  # Clear progress message
            response.raise_for_status()
            data = response.json()
            answer = data.get("answer", "No answer received")
            
            # Check if answer indicates no data found
            answer_lower = answer.lower()
            no_data_indicators = [
                "couldn't find",
                "no relevant",
                "no information",
                "don't have",
                "not found"
            ]
            is_no_data = any(indicator in answer_lower for indicator in no_data_indicators)
            
            if is_no_data:
                st.warning("⚠️ No Data Found")
                st.info(answer)
                st.caption("The system couldn't find relevant information in the dataset to answer this question.")
            else:
                st.success("✅ Answer:")
                st.info(answer)
            
            # Store in session for history
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.insert(0, {
                "question": question, 
                "answer": answer,
                "no_data": is_no_data
            })
                
        except requests.exceptions.Timeout:
            progress_placeholder.empty()
            st.error("⏱️ Request timed out after 120 seconds. The system is loading messages and computing embeddings. Please try again in a moment.")
            st.info("💡 Tip: The first request may take 60-120 seconds (downloads models, fetches messages, computes embeddings). Subsequent requests are much faster.")
        except requests.exceptions.RequestException as e:
            progress_placeholder.empty()
            st.error(f"Error connecting to API: {str(e)}")
            st.info("Make sure the API server is running at " + API_URL)
        except Exception as e:
            progress_placeholder.empty()
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a question")

# Show history
if "history" in st.session_state and st.session_state.history:
    st.divider()
    st.subheader("Recent Questions")
    for i, item in enumerate(st.session_state.history[:5]):  # Show last 5
        is_no_data = item.get("no_data", False)
        status_icon = "⚠️" if is_no_data else "✅"
        with st.expander(f"{status_icon} Q: {item['question']}"):
            if is_no_data:
                st.warning("**Answer:** " + item['answer'])
                st.caption("No relevant data found in the dataset")
            else:
                st.write(f"**Answer:** {item['answer']}")

# Footer
st.divider()
st.markdown("**API Endpoint:** " + API_URL)
try:
    health_check = requests.get(f"{API_URL}/health", timeout=5)
    status = "🟢 Online" if health_check.status_code == 200 else "🔴 Offline"
    
    # Get system status
    try:
        status_response = requests.get(f"{API_URL}/status", timeout=5)
        if status_response.status_code == 200:
            status_data = status_response.json()
            st.markdown(f"**Status:** {status}")
            st.caption(f"Mode: {status_data.get('mode', 'unknown')} | "
                      f"Embeddings: {'Ready' if status_data.get('embeddings_ready') else 'Not ready'} | "
                      f"SLM: {'Ready' if status_data.get('slm_ready') else 'Not ready'}")
        else:
            st.markdown("**Status:** " + status)
    except:
        st.markdown("**Status:** " + status)
except:
    status = "🔴 Offline"
    st.markdown("**Status:** " + status)

