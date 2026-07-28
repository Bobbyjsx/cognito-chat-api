# Cognito-Chat API

Cognito-Chat is a modern, high-performance API backend built with FastAPI, Firebase Firestore, and the Google Antigravity SDK. It provides an intelligent conversational agent with built-in token tracking and JWT authentication.

## Features

- **FastAPI**: Extremely fast asynchronous API.
- **Firebase Firestore**: Scalable NoSQL document database.
- **Antigravity SDK**: Integration with Google's latest AI models.
- **JWT Authentication**: Full-scale auth with secure signup, login, and password hashing (`bcrypt`).
- **Token Limits**: Tracks and enforces maximum token limits for each user.
- **Docker Ready**: Designed to be easily containerized and deployed.

## Requirements

- Python 3.11+
- Firebase Service Account Credentials

## Setup

1. **Clone the repository**

2. **Create a virtual environment and install dependencies**
   ```bash
   make install
   ```

3. **Configure Environment Variables**
   Rename `.env.example` to `.env` and fill in the required keys:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   FIREBASE_CREDENTIALS_PATH=firebase-credentials.json
   ```

4. **Firebase Credentials**
   Download your Firebase Service Account JSON key from the Firebase Console and save it as `firebase-credentials.json` in the project root.

## Running the Application

To start the development server with auto-reload:

```bash
make run
```
The API will be available at `http://127.0.0.1:8000`. You can access the Swagger UI documentation at `http://127.0.0.1:8000/docs`.

## Development Commands

We use `ruff` for fast linting and formatting.

- **Run Linter**: `make lint`
- **Fix Lint Issues**: `make lint-fix`
- **Format Code**: `make format`
- **Run Tests**: `make test`
