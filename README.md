# Deep Learning Toolbox

An interactive FastAPI + static frontend toolbox for perceptron/MLP classification, activation functions, gradient descent, model diagnosis, what-if simulation, OpenCV vision, and RNN next-word prediction.

## Project Structure

```text
.
├── backend/
│   ├── main.py
│   ├── schemas.py
│   ├── models/
│   │   ├── activation.py
│   │   ├── datasets.py
│   │   ├── explainability.py
│   │   ├── gradient_descent.py
│   │   ├── perceptron_mlp.py
│   │   ├── rnn.py
│   │   └── vision.py
│   ├── routers/
│   │   ├── activation.py
│   │   ├── explainability.py
│   │   ├── gradient_descent.py
│   │   ├── perceptron.py
│   │   ├── rnn.py
│   │   └── vision.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── dashboard.html
│   ├── style.css
│   ├── js/
│   │   ├── api.js
│   │   ├── charts.js
│   │   ├── navigation.js
│   │   ├── perceptron-view.js
│   │   ├── activation-view.js
│   │   ├── gradient-view.js
│   │   ├── explain-view.js
│   │   ├── what-if-view.js
│   │   ├── vision-view.js
│   │   ├── rnn-view.js
│   │   └── dashboard.js
│   └── vercel.json
└── render.yaml
```

## Run Locally

Backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Frontend:

Open `frontend/index.html` directly, or serve the folder:

```bash
cd frontend
python -m http.server 8001
```

Dashboard URL when served locally:

```text
http://127.0.0.1:8001/dashboard.html
```

## Deploy

Backend on Render:

Use the included `render.yaml`, or create a Python web service with:

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Frontend on Vercel:

Deploy the `frontend/` folder as a static project. In the dashboard, set the Backend URL field to your Render URL, for example:

```text
https://your-service.onrender.com
```

## API

- `GET /api/health`
- `GET /api/perceptron/datasets`
- `POST /api/perceptron/train`
- `POST /api/activation/curve`
- `POST /api/gradient-descent/simulate`
- `POST /api/insights/explain`
- `POST /api/insights/what-if`
- `POST /api/vision/analyze`
- `POST /api/rnn/predict`

The RNN endpoint accepts text such as `Hello how are` and returns the predicted next word plus top alternatives and a hidden-state preview.
