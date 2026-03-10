# GeominerAI — Complete Beginner Setup Guide

This guide assumes you know nothing about programming. Follow it step by step.

---

## Part 1: Software You Need to Install

Install these 5 programs on your computer **before** doing anything else:

### 1. VS Code (Code Editor)
- Go to: https://code.visualstudio.com/download
- Download the version for your operating system (Windows / Mac / Linux)
- Run the installer, click "Next" through everything
- This is where you'll open and edit the project

### 2. Python 3.11 or newer
- Go to: https://www.python.org/downloads/
- Click the big yellow "Download Python" button
- Run the installer
- **IMPORTANT (Windows):** At the very first screen of the installer, check the box that says **"Add Python to PATH"** — if you miss this, nothing will work later
- Click "Install Now"
- To verify: open a terminal and type `python --version` — you should see something like `Python 3.11.x`

### 3. Node.js 20 or newer
- Go to: https://nodejs.org/
- Download the **LTS** version (the green button on the left)
- Run the installer, click "Next" through everything
- To verify: open a terminal and type `node --version` — you should see something like `v20.x.x`

### 4. Git
- Go to: https://git-scm.com/downloads
- Download for your operating system
- Run the installer, use all default settings
- To verify: open a terminal and type `git --version`

### 5. Docker Desktop
- Go to: https://www.docker.com/products/docker-desktop/
- Download for your operating system
- Run the installer
- **You may need to restart your computer after installing**
- Open Docker Desktop and wait until it says "Docker Desktop is running" (the whale icon in your taskbar should stop animating)

---

## Part 2: Get Your HuggingFace Token

The project uses HuggingFace's AI service to power the language model. You need a free token (like a password) to access it.

**If you already have a HuggingFace token** (starts with `hf_...`), you can use the same one. No need to create a new one.

**If you don't have one yet:**

1. Go to https://huggingface.co/settings/tokens
2. Click "Sign Up" if you don't have an account (it's free)
3. Once logged in, click **"New token"**
4. Name it anything you want (e.g., `geominerai`)
5. Select **"Read"** access
6. Click **"Generate"**
7. Copy the token — it looks like `hf_AbCdEfGhIjKlMnOpQrStUv`
8. **Save it somewhere safe** (you'll need it soon)

---

## Part 3: Choose How to Run the App

You have 3 options. Pick the one that matches your comfort level:

| Option | Difficulty | Best For |
|--------|-----------|----------|
| **A: VS Code (3 terminals)** | Medium | Development, seeing how everything works |
| **B: Docker (fewest steps)** | Easy | Just want it running quickly |
| **C: Streamlit only** | Easiest | Just want to try the original app, no Docker needed |

---

## Option A: Run Through VS Code (Recommended)

This gives you the full new architecture: React frontend + FastAPI backend + database.

### Step 1: Open the project in VS Code

1. Open **VS Code**
2. Click **File** in the top menu bar
3. Click **Open Folder**
4. Navigate to the `geominerAI_GeoGPT` folder on your computer
5. Click **"Select Folder"** (or "Open" on Mac)

You should now see all the project files in the left sidebar.

### Step 2: Open a terminal inside VS Code

1. Click **Terminal** in the top menu bar
2. Click **New Terminal**
3. A terminal panel appears at the bottom of VS Code
4. This is where you'll type commands

### Step 3: Start Docker Desktop

1. Find and open the **Docker Desktop** application
2. Wait until it's fully running (whale icon in your taskbar stops animating)
3. You can minimize Docker Desktop — it just needs to be running in the background

### Step 4: Start the database and cache

In the VS Code terminal, type this command and press Enter:

```
docker-compose up postgres redis
```

**What this does:** Starts a PostgreSQL database and Redis cache inside Docker containers.

Wait until you see a message like:
```
postgres-1  | database system is ready to accept connections
```

**DO NOT close this terminal. Leave it running.**

### Step 5: Open a second terminal

1. In the terminal panel at the bottom of VS Code, look for a **+** icon (plus sign)
2. Click it to open a new terminal tab
3. You now have two terminals — you can switch between them using the tabs

### Step 6: Set up the backend (Python)

In the **second terminal**, type these commands **one at a time**, pressing Enter after each:

**Create a virtual environment:**
```
python -m venv .venv
```

**Activate the virtual environment:**

On Mac or Linux:
```
source .venv/bin/activate
```

On Windows (PowerShell):
```
.venv\Scripts\activate
```

You should see `(.venv)` appear at the start of your terminal line. This means it's active.

**Install the backend dependencies:**
```
pip install -r backend/requirements.txt
```

This will download and install a lot of packages. Wait for it to finish (may take 2-5 minutes).

**Set your HuggingFace token:**

On Mac or Linux:
```
export HF_TOKEN=hf_YOUR_TOKEN_HERE
```

On Windows (PowerShell):
```
$env:HF_TOKEN="hf_YOUR_TOKEN_HERE"
```

Replace `hf_YOUR_TOKEN_HERE` with the actual token you copied from HuggingFace.

**Start the backend server:**
```
uvicorn backend.main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**DO NOT close this terminal. Leave it running.**

### Step 7: Open a third terminal and start the frontend

1. Click the **+** icon in the terminal panel again (third terminal)
2. Type these commands one at a time:

```
cd frontend
```

```
npm install
```

This installs frontend dependencies (may take 1-3 minutes the first time).

```
npm run dev
```

You should see:
```
  VITE v6.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
```

### Step 8: Open the app in your browser!

1. Open your web browser (Chrome, Firefox, or Edge)
2. Type this in the address bar and press Enter:

```
http://localhost:5173
```

3. You should see the GeominerAI interface with a map, chat panel, and layer panel!

### To stop everything when you're done:

1. Go to the **third terminal** (frontend) → press `Ctrl+C`
2. Go to the **second terminal** (backend) → press `Ctrl+C`
3. Go to the **first terminal** (Docker) → press `Ctrl+C`, then type:
```
docker-compose down
```

---

## Option B: Run Everything with Docker (Fewer Steps)

This runs the backend, database, and cache all inside Docker. You only manually run the frontend.

### Step 1: Make sure Docker Desktop is running

Open Docker Desktop and wait for it to be ready.

### Step 2: Open VS Code and open a terminal

1. Open VS Code → File → Open Folder → select `geominerAI_GeoGPT`
2. Terminal → New Terminal

### Step 3: Start all backend services

On Mac or Linux:
```
export HF_TOKEN=hf_YOUR_TOKEN_HERE
docker-compose up
```

On Windows (PowerShell):
```
$env:HF_TOKEN="hf_YOUR_TOKEN_HERE"
docker-compose up
```

**What this does:** Starts the API server, database, cache, and background worker all at once.

Wait until you see healthy log output from all services. The first time may take 5-10 minutes as Docker downloads images.

**Leave this terminal running.**

### Step 4: Open a second terminal and start the frontend

Click **+** in the terminal panel, then type:

```
cd frontend
```
```
npm install
```
```
npm run dev
```

### Step 5: Open the app

Go to **http://localhost:5173** in your browser.

### To stop:
1. `Ctrl+C` in the frontend terminal
2. `Ctrl+C` in the Docker terminal, then type `docker-compose down`

---

## Option C: Run the Original Streamlit App (Simplest)

This runs the original single-file app. **No Docker needed. No Node.js needed. Just Python.**

### Step 1: Open VS Code and a terminal

1. Open VS Code → File → Open Folder → select `geominerAI_GeoGPT`
2. Terminal → New Terminal

### Step 2: Set up Python environment

**Create a virtual environment:**
```
python -m venv .venv
```

**Activate it:**

On Mac or Linux:
```
source .venv/bin/activate
```

On Windows (PowerShell):
```
.venv\Scripts\activate
```

### Step 3: Install dependencies

```
pip install -r requirements.txt
```

### Step 4: Set your token

On Mac or Linux:
```
export HF_TOKEN=hf_YOUR_TOKEN_HERE
```

On Windows (PowerShell):
```
$env:HF_TOKEN="hf_YOUR_TOKEN_HERE"
```

### Step 5: Run the app

```
streamlit run app.py
```

Your browser should open automatically to **http://localhost:8501** with the GeominerAI app!

### To stop:
Press `Ctrl+C` in the terminal.

---

## Quick Reference: What Runs Where

| Service | URL | What It Does |
|---------|-----|-------------|
| **Frontend (React)** | http://localhost:5173 | The map interface you interact with |
| **Backend API** | http://localhost:8000 | Processes your requests, runs the AI |
| **API Documentation** | http://localhost:8000/docs | Interactive docs to test the API |
| **PostgreSQL Database** | localhost:5432 | Stores your data, layers, chat history |
| **Redis Cache** | localhost:6379 | Caching and background task queue |
| **Streamlit (Option C)** | http://localhost:8501 | The original single-page app |

---

## Troubleshooting

### "python not found" or "python3 not found"
- **Windows:** Reinstall Python from python.org. Make sure to check **"Add Python to PATH"** during installation. Restart VS Code after installing.
- **Mac:** Try typing `python3` instead of `python`. If that works, use `python3` everywhere this guide says `python`.

### "npm not found"
- Reinstall Node.js from nodejs.org
- Restart VS Code (close it completely and reopen)

### "docker-compose not found"
- Make sure Docker Desktop is open and running
- Try `docker compose up` (with a space, no hyphen) instead of `docker-compose up` — newer versions of Docker use this format

### "pip install" fails with errors about compiling
- **Windows:** Install "Visual C++ Build Tools" from https://visualstudio.microsoft.com/visual-cpp-build-tools/
- **Mac:** Open terminal and run: `xcode-select --install`

### Backend says "connection refused" to database
- The database (Docker) isn't running yet. Go back to Step 4 and make sure `docker-compose up postgres redis` is running in its terminal.

### Frontend shows a blank white page
- Check that the backend is running on port 8000 (you should see the Uvicorn message)
- Open browser developer tools (press F12) and check the Console tab for errors

### "Address already in use" error
- Another program is using that port. Either close that program, or change the port:
  - Backend: `uvicorn backend.main:app --reload --port 8001`
  - Frontend: Edit `frontend/vite.config.ts` and change the port number

### HuggingFace token errors
- Go to https://huggingface.co/settings/tokens and verify your token is valid
- Make sure you copied the full token (it starts with `hf_`)
- Make sure there are no extra spaces around the token

### Everything seems to work but AI responses fail
- Your HuggingFace token might not have access to the model. Go to https://huggingface.co/HuggingFaceH4/zephyr-7b-beta and click "Agree" to the model's terms if prompted.

---

## Next Time You Want to Run It

After the first-time setup, you don't need to install things again. Just:

1. Open Docker Desktop
2. Open VS Code → Open the project folder
3. Open a terminal and run `docker-compose up postgres redis`
4. Open a second terminal:
   ```
   source .venv/bin/activate
   export HF_TOKEN=hf_YOUR_TOKEN_HERE
   uvicorn backend.main:app --reload --port 8000
   ```
5. Open a third terminal:
   ```
   cd frontend
   npm run dev
   ```
6. Go to http://localhost:5173

That's it! No need to reinstall anything.
