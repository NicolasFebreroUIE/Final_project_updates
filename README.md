# ALIS-CORE: Algorithmic Impartiality Paradox

**Author:** Nicolás Febrero Lubián
**Degree:** Grado en Ingeniería en Sistemas Inteligentes
**University:** Universidad Intercontinental de la Empresa (UIE), Vigo, Galicia

## Description
ALIS-CORE is a multimodal forensic AI system demonstrating the vulnerabilities of algorithmic justice to linguistic surface forms, integrating NLP (Retrieval-Augmented Generation), Computer Vision (Emotion & Micro-expression analysis), and a neuro-symbolic logic engine.

## Prerequisites
- Python 3.9+ (64-bit)
- A dedicated GPU with CUDA support is highly recommended for optimal real-time performance.
- Git (optional, for cloning purposes)

## Installation & Setup

1. **Open a terminal (PowerShell)** in this project's root directory (`Final_Project_Nicolas_Febrero`).

2. **Create a virtual environment**:
   ```powershell
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   *(If you encounter execution policy errors, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first).*

4. **Install all required dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

## Running the Application

To ensure no previous background instances of the application are conflicting with the port, and to launch the web application interface correctly, execute the exact following command in PowerShell:

```powershell
Get-Process python* -ErrorAction SilentlyContinue | Stop-Process -Force; .\.venv\Scripts\python web/app.py
```

After execution, the system will start the Flask server. Open your web browser and navigate to:
**http://127.0.0.1:5000**
