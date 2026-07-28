import pytest

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_signup(client):
    response = client.post(
        "/auth/signup",
        json={"email": "newuser@example.com", "password": "newpassword123"}
    )
    # Even though it's a mock, it will simulate a successful creation
    print("SIGNUP RESPONSE:", response.json())
    assert response.status_code == 201
    assert response.json()["email"] == "newuser@example.com"

def test_login(client):
    # Create the user first
    client.post(
        "/auth/signup",
        json={"email": "testuser@example.com", "password": "securepassword123"}
    )
    
    response = client.post(
        "/auth/login",
        data={"username": "testuser@example.com", "password": "securepassword123"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_get_me(client):
    # Create the user first
    client.post(
        "/auth/signup",
        json={"email": "testuser@example.com", "password": "securepassword123"}
    )
    
    # First login to get a token
    login_response = client.post(
        "/auth/login",
        data={"username": "testuser@example.com", "password": "securepassword123"}
    )
    token = login_response.json()["access_token"]
    
    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["email"] == "testuser@example.com"

def test_chat(client, mock_agent):
    # Create the user first
    client.post(
        "/auth/signup",
        json={"email": "testuser@example.com", "password": "securepassword123"}
    )
    
    # First login to get a token
    login_response = client.post(
        "/auth/login",
        data={"username": "testuser@example.com", "password": "securepassword123"}
    )
    token = login_response.json()["access_token"]
    
    response = client.post(
        "/agent/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"message": "Hello!"}
    )
    
    assert response.status_code == 200
    assert "session_id" in response.json()
    assert response.json()["response"] == "Hello from mocked agent!"
