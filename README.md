# Features

## Secretary Chat & Employee Grade

- Secretary Chat: AI interface for recommendations, messaging, and business analysis, all in a conversational chat format.
- Employee Grade: On the right side of the chat, you can check the performance level of any employee (bronze/silver/gold) based on backend data. Enter the employee ID and instantly see their level, sales, and what is needed for the next level.

# Installation

To install the required Python packages for the backend, run:


```powershell
pip install -r requirements.txt
```

Make sure you are using the correct Python environment.

# OpenAI Business Assistant - How to Run

## Backend (API)

1. Start the FastAPI backend (from the OpenAI folder):
   ```sh
   uvicorn api_secretary:app --reload --port 1919
   ```

## Frontend

### Chat UI (Recommended)
- Open `frontend_secretary_chat.html` in your browser (Chrome/Edge).
- All interactions are done in a conversational chat format.

### Classic UI (with dropdowns)
- Open `frontend_secretary_full.html` in your browser for the full interface with classic selection options.


---

## Important Requirement

For the project to work correctly, there must be a database containing all sales, and the application must be connected to this database.

