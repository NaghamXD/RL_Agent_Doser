.\.venv\Scripts\Activate.ps1
python app.py --debug
python -m http.server 8000
http://localhost:8000/dashboard.html



# RL Agent - Radiation Therapy Planning UI

A modern, interactive web dashboard for visualizing and monitoring RL Agent-based radiation therapy treatment planning.

## ðŸŽ¯ Features

### Dashboard Components
- **Patient Selection** - Choose from available patients or select randomly
- **PTV Coverage Tracking** - Visual progress for 3 Planning Target Volumes (PTV70, PTV63, PTV56)
  - Shows prescriptions vs. current coverage
  - Color-coded status (green/orange/red based on coverage %)
  - Real-time progress bars

- **OAR Monitoring** - Track 5 Organs At Risk (Brainstem, SpinalCord, Mandible, LeftParotid, RightParotid)
  - Mean and max dose visualization
  - Tolerance limits comparison
  - Violation detection alerts

- **Fraction-by-Fraction Progress**
  - Interactive timeline showing all fractions
  - Real-time reward tracking
  - Fraction selection for detailed metrics

- **Dose Maps**
  - Interactive heatmap visualization per beam
  - 9 beam selection interface
  - Color-coded dose intensity display

- **Agent Performance Metrics**
  - Reward trends across fractions
  - Action statistics (mean/max)
  - OAR penalty vs. PTV reward breakdown

- **Real-time Monitoring**
  - System status indicator
  - Treatment progress tracker
  - Agent state visualization

## ðŸ“‹ Prerequisites

- Python 3.8+
- The existing RL Agent codebase with trained models
- A modern web browser (Chrome, Firefox, Safari, Edge)

## ðŸš€ Quick Start

### 1. Install Dependencies

```bash
# Install UI-specific requirements
pip install -r requirements.txt
```

### 2. Start the Backend Server

```bash
# Basic usage - uses default config and best.pt model
python app.py

# Or with custom parameters
python app.py --config configs/default.yaml --ckpt runs/best.pt --port 5000 --debug
```

Options:
- `--config`: Path to configuration file (default: `configs/default.yaml`)
- `--ckpt`: Path to trained model checkpoint (default: `runs/best.pt`)
- `--split`: Dataset split to use (default: `validation`)
- `--port`: Port to run Flask server on (default: `5000`)
- `--debug`: Enable Flask debug mode

**Expected Output:**
```
Initializing backend...
Backend ready!
 * Running on http://0.0.0.0:5000
```

### 3. Open the Frontend

Open your browser and navigate to:
```
file:///path/to/RL_Agent/ui.html
```

Or serve it via a simple HTTP server:
```bash
# Python 3
python -m http.server 8000 --directory .

# Then open: http://localhost:8000/ui.html
```

## ðŸ“– Usage Guide

### Main Workflow

1. **Select a Patient**
   - Use the dropdown to select from available patients
   - Click "ðŸŽ² Random Patient" to pick a random patient

2. **Run Simulation**
   - Click "â–¶ï¸ Run Simulation" to start the agent evaluation
   - The system will simulate up to 5 fractions for the selected patient
   - Progress will be displayed as the simulation runs

3. **Explore Results**
   - **Fraction Timeline**: Click on any fraction badge to view detailed metrics
   - **PTV Coverage**: Monitor coverage percentage for each target volume
   - **OAR Monitoring**: Check organ dose constraints
   - **Dose Maps**: Select different beams to see dose distribution
   - **Performance Metrics**: View reward components and action statistics

### Dashboard Sections

#### Control Panel (Top)
- Filter and select patients
- Launch simulations
- Add custom parameters

#### PTV Coverage Card
- Shows dose delivery progress for each target
- Color indicators:
  - ðŸŸ¢ Green: >95% coverage (excellent)
  - ðŸŸ¡ Orange: 80-95% coverage (acceptable)
  - ðŸ”´ Red: <80% coverage (underdose)

#### OAR Monitoring Card
- Lists each organ with current dose
- Compares against tolerance limits
- Shows violation status (âœ“ OK / âŒ VIOLATION)

#### Fraction Metrics Card
- Total reward for current fraction
- OAR penalty component
- PTV reward component
- Max beamlet intensity used

#### Reward Trend Chart
- Line graph showing reward progression across fractions
- Helps identify treatment plan stability

#### Beam Selection & Dose Map
- Choose any of 9 beams to visualize
- Heatmap shows dose distribution
- Interactive beam buttons

#### Agent Performance Summary
- Mean action (average beamlet intensity)
- Max action (peak intensity)
- Total fractions evaluated
- Average reward across all fractions

## ðŸ”Œ API Endpoints

The Flask backend provides the following REST APIs:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/config` | GET | System configuration |
| `/api/patients` | GET | List available patients |
| `/api/patients/random` | GET | Get random patient ID |
| `/api/patients/<id>/data` | GET | Get patient anatomy data |
| `/api/patients/<id>/evaluation` | GET | Get pre-computed evaluation results |
| `/api/patients/<id>/simulate` | POST | Run live agent simulation |
| `/api/status` | GET | System status and metrics |

### Example API Calls

```bash
# Get available patients
curl http://localhost:5000/api/patients

# Get random patient
curl http://localhost:5000/api/patients/random

# Get system config
curl http://localhost:5000/api/config

# Run simulation for pt_201
curl -X POST http://localhost:5000/api/patients/pt_201/simulate
```

## ðŸŽ¨ UI Design Features

- **Modern Gradient Theme**: Blue-Purple primary colors with semantic color coding
- **Responsive Grid Layout**: Automatically adapts to desktop/tablet/mobile
- **Real-time Charts**: Interactive Chart.js visualizations
- **Smooth Animations**: Fade-in effects and hover interactions
- **Accessible Design**: Focus states, color contrast, proper semantics
- **Dark-aware**: Supports system preferences (optional enhancement)

### Color Palette
- Primary: Blue (#3b82f6)
- Secondary: Green (#10b981)
- Danger: Red (#ef4444)
- Warning: Amber (#f59e0b)
- Neutral: Gray (#1f2937, #f3f4f6)

## ðŸ”§ Architecture

### Backend (`app.py`)
- Flask REST API server
- Patient data management
- Agent simulation orchestration
- Real-time model evaluation

### Frontend (`ui.html`)
- React-based single-page application (using CDN)
- Responsive dashboard layout
- Chart.js visualizations
- Interactive components

### Data Flow
```
UI (React) 
    â†“ (HTTP REST)
Flask API (app.py)
    â†“ (Python API)
DoseEnv (dose_env.py)
    â†“
PPO Agent (ppo.py)
    â†“
Dose Influence Matrix
    â†“
Patient Treatment Plan
```

## ðŸ“Š Supported Structures

### Planning Target Volumes (PTVs)
- **PTV70**: 70 Gy prescription
- **PTV63**: 63 Gy prescription
- **PTV56**: 56 Gy prescription

### Organs At Risk (OARs)
- **Brainstem**: 54 Gy tolerance (serial organ)
- **SpinalCord**: 45 Gy tolerance (serial organ)
- **Mandible**: 70 Gy tolerance
- **LeftParotid**: 26 Gy tolerance
- **RightParotid**: 26 Gy tolerance

## ðŸ› Troubleshooting

### Backend Won't Start
```
Error: Port 5000 already in use
â†’ Use --port flag: python app.py --port 5001
```

### CORS Error in Browser Console
```
Error: Access to XMLHttpRequest blocked by CORS policy
â†’ CORS is already enabled in app.py. Ensure backend is running.
```

### Can't Connect to Backend
```
Error: Failed to load data
â†’ Check if Flask server is running on http://localhost:5000
â†’ Try: curl http://localhost:5000/api/health
```

### No Patients Visible
```
â†’ Ensure processed data exists in: data/processed/validation/
â†’ Verify config split path is correct
â†’ Run preprocessing script first: python scripts/preprocess.py
```

### Simulation Takes Too Long
```
â†’ Reduce number of fractions simulated in app.py (currently 5)
â†’ Use faster GPU if available: check --device flag
â†’ Close other applications
```

## ðŸ“ˆ Performance Metrics

- **Frontend Load**: ~2-3 seconds (initial data fetch)
- **Simulation Time**: ~1-2 seconds per fraction (depends on hardware)
- **Chart Rendering**: <500ms
- **API Response**: <100ms (except simulation)

## ðŸ” Security Notes

- No authentication implemented (add if exposing to network)
- CORS enabled for localhost only (configure for production)
- No data persistence to disk
- Ensure access to sensitive patient data is restricted

## ðŸŽ“ Development & Customization

### Adding New Metrics

Edit the `Dashboard` component in `ui.html` to add new cards:

```jsx
<div className="card">
    <div className="card-title">
        <span className="card-icon">ðŸ“Š</span>
        New Metric
    </div>
    {/* Content here */}
</div>
```

### Modifying Color Scheme

Update CSS variables in `ui.html`:

```css
:root {
    --primary: #your-color;
    --secondary: #your-color;
    /* etc */
}
```

### Adding More Visualizations

Integrate additional libraries:
- Plotly.js: Already included, use `Plotly.newPlot()`
- D3.js: For custom visualizations
- Three.js: For 3D dose distributions

### Extending Backend API

Add new endpoints in `app.py`:

```python
@app.route('/api/your-endpoint', methods=['GET'])
def your_endpoint():
    return jsonify({'data': 'value'})
```

## ðŸ“ File Structure

```
RL_Agent/
â”œâ”€â”€ app.py                  # Flask backend API
â”œâ”€â”€ ui.html                 # React frontend dashboard
â”œâ”€â”€ requirements.txt     # Python dependencies for UI
â”œâ”€â”€ UI_README.md           # This file
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ env/
â”‚   â”‚   â”œâ”€â”€ dose_env.py    # Environment definition
â”‚   â”‚   â””â”€â”€ reward.py      # Reward calculation
â”‚   â”œâ”€â”€ agents/
â”‚   â”‚   â””â”€â”€ ppo.py         # PPO agent implementation
â”‚   â””â”€â”€ config.py          # Configuration
â”œâ”€â”€ runs/
â”‚   â”œâ”€â”€ best.pt            # Best trained model
â”‚   â””â”€â”€ eval_*.txt         # Evaluation results
â””â”€â”€ data/
    â””â”€â”€ processed/         # Preprocessed patient data
```

## ðŸš€ Future Enhancements

- [ ] 3D dose visualization (Three.js)
- [ ] Dose-Volume Histogram (DVH) charts
- [ ] Real-time plan optimization
- [ ] Comparison mode (multiple agents/plans)
- [ ] Export treatment plans (DICOM)
- [ ] User authentication & patient security
- [ ] Database backend for result persistence
- [ ] Batch patient evaluation
- [ ] Custom reward function editor
- [ ] Integration with TPS software

## ðŸ“ž Support

For issues or questions:
1. Check the Troubleshooting section
2. Review Flask server logs: `python app.py --debug`
3. Open browser developer tools (F12) for JavaScript errors
4. Check API connectivity: `curl http://localhost:5000/api/health`

## ðŸ“„ License

Same as parent RL Agent project.

---

**Last Updated**: June 2026
**Version**: 1.0
