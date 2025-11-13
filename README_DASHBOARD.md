# Streamlit Dashboard

A simple web interface for the Member Data QA System.

## Run Locally

```bash
# Install streamlit (if not already installed)
pip install streamlit

# Run the dashboard
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Features

- Simple text input for questions
- Example question buttons in sidebar
- Answer display
- Question history
- API status indicator

## Configuration

The dashboard connects to the deployed API at:
`https://aurora-applied-ai-ml-engineer-take-home.onrender.com`

To change the API URL, edit the `API_URL` variable in `dashboard.py`.

