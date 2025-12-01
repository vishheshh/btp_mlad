# Starting the Power Grid Protection System

## Quick Start

You can start both the API server and Streamlit dashboard together using one of these methods:

### Option 1: Python Script (Recommended)

```bash
python start_services.py
```

This starts both services and manages them together. Press CTRL+C to stop both.

### Option 2: Windows Batch File

```bash
start_services.bat
```

This opens both services in separate command windows. Close the windows to stop them.

### Option 3: PowerShell Script

```powershell
.\start_services.ps1
```

This opens both services in separate PowerShell windows. Close the windows to stop them.

### Option 4: Manual Start (Separate Terminals)

**Terminal 1 - API Server:**

```bash
python api/api_server.py
```

**Terminal 2 - Streamlit Dashboard:**

```bash
python -m streamlit run frontend/streamlit_dashboard.py
```

## Access Points

- **Dashboard**: http://localhost:8501
- **API**: http://localhost:5000/api/

## Notes

- The API server must start before the Streamlit dashboard
- The scripts automatically wait 3 seconds for the API server to initialize
- Make sure you're in the `power_grid_protection` directory when running these scripts
