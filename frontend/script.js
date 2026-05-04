const form = document.getElementById("controlForm");
const canvas = document.getElementById("decisionCanvas");

if (form && canvas) {
    const controls = {
        apiBase: document.getElementById("apiBase"),
        dataset: document.getElementById("dataset"),
        modelType: document.getElementById("modelType"),
        trainToggle: document.getElementById("trainToggle"),
        w1: document.getElementById("w1"),
        w2: document.getElementById("w2"),
        bias: document.getElementById("bias"),
        learningRate: document.getElementById("learningRate"),
        epochs: document.getElementById("epochs"),
    };

    const outputs = {
        w1: document.getElementById("w1Value"),
        w2: document.getElementById("w2Value"),
        bias: document.getElementById("biasValue"),
        learningRate: document.getElementById("learningRateValue"),
        apiStatus: document.getElementById("apiStatus"),
        conceptText: document.getElementById("conceptText"),
        accuracyMetric: document.getElementById("accuracyMetric"),
        datasetMetric: document.getElementById("datasetMetric"),
        modelMetric: document.getElementById("modelMetric"),
        historyMetric: document.getElementById("historyMetric"),
        resultBadge: document.getElementById("resultBadge"),
        truthTable: document.getElementById("truthTable"),
        weightControls: document.getElementById("weightControls"),
    };

    const storedApiBase = localStorage.getItem("dlt_api_base");
    if (storedApiBase) {
        controls.apiBase.value = storedApiBase;
    }

    const syncSliderLabels = () => {
        outputs.w1.value = Number(controls.w1.value).toFixed(2);
        outputs.w2.value = Number(controls.w2.value).toFixed(2);
        outputs.bias.value = Number(controls.bias.value).toFixed(2);
        outputs.learningRate.value = Number(controls.learningRate.value).toFixed(2);
    };

    const setStatus = (message, type = "") => {
        outputs.apiStatus.textContent = message;
        outputs.apiStatus.className = `api-status ${type}`.trim();
    };

    const setBadge = (message, type = "") => {
        outputs.resultBadge.textContent = message;
        outputs.resultBadge.className = `result-badge ${type}`.trim();
    };

    const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
    const graphMin = -0.25;
    const graphMax = 1.25;
    const graphSpan = graphMax - graphMin;

    const toCanvasPoint = (x1, x2, width, height, padding) => ({
        x: padding + ((x1 - graphMin) / graphSpan) * (width - padding * 2),
        y: height - padding - ((x2 - graphMin) / graphSpan) * (height - padding * 2),
    });

    const drawGridLines = (ctx, width, height, padding) => {
        ctx.strokeStyle = "hsl(214, 45%, 90%)";
        ctx.lineWidth = 1;
        for (let tick = -0.25; tick <= 1.2501; tick += 0.25) {
            const vertical = toCanvasPoint(tick, graphMin, width, height, padding);
            ctx.beginPath();
            ctx.moveTo(vertical.x, padding);
            ctx.lineTo(vertical.x, height - padding);
            ctx.stroke();

            const horizontal = toCanvasPoint(graphMin, tick, width, height, padding);
            ctx.beginPath();
            ctx.moveTo(padding, horizontal.y);
            ctx.lineTo(width - padding, horizontal.y);
            ctx.stroke();
        }

        ctx.strokeStyle = "hsl(218, 28%, 24%)";
        ctx.lineWidth = 1.4;
        const xAxisStart = toCanvasPoint(graphMin, 0, width, height, padding);
        const xAxisEnd = toCanvasPoint(graphMax, 0, width, height, padding);
        ctx.beginPath();
        ctx.moveTo(xAxisStart.x, xAxisStart.y);
        ctx.lineTo(xAxisEnd.x, xAxisEnd.y);
        ctx.stroke();

        const yAxisStart = toCanvasPoint(0, graphMin, width, height, padding);
        const yAxisEnd = toCanvasPoint(0, graphMax, width, height, padding);
        ctx.beginPath();
        ctx.moveTo(yAxisStart.x, yAxisStart.y);
        ctx.lineTo(yAxisEnd.x, yAxisEnd.y);
        ctx.stroke();

        ctx.fillStyle = "hsl(216, 13%, 48%)";
        ctx.font = "600 13px Inter, system-ui, sans-serif";
        ctx.fillText("x1", width - padding + 10, xAxisEnd.y + 4);
        ctx.fillText("x2", yAxisEnd.x - 8, padding - 12);
    };

    const drawDecisionCanvas = (result) => {
        const ctx = canvas.getContext("2d");
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const width = Math.max(360, Math.floor(rect.width));
        const height = Math.max(360, Math.floor(rect.height));
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const padding = width < 560 ? 42 : 58;
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, width, height);

        if (Array.isArray(result.grid) && result.grid.length > 0) {
            const steps = Math.round(Math.sqrt(result.grid.length));
            const cellWidth = (width - padding * 2) / steps;
            const cellHeight = (height - padding * 2) / steps;
            result.grid.forEach((cell) => {
                const point = toCanvasPoint(cell.x1, cell.x2, width, height, padding);
                ctx.fillStyle = cell.prediction === 1
                    ? "hsla(151, 66%, 42%, 0.14)"
                    : "hsla(39, 92%, 51%, 0.16)";
                ctx.fillRect(point.x - cellWidth / 2, point.y - cellHeight / 2, cellWidth + 1, cellHeight + 1);
            });
        }

        drawGridLines(ctx, width, height, padding);

        if (result.boundary) {
            ctx.strokeStyle = "hsl(214, 82%, 44%)";
            ctx.lineWidth = 3;
            ctx.setLineDash([]);
            ctx.beginPath();
            if (result.boundary.type === "vertical") {
                const x = toCanvasPoint(result.boundary.x1, 0, width, height, padding).x;
                ctx.moveTo(x, padding);
                ctx.lineTo(x, height - padding);
            } else {
                const start = result.boundary.points[0];
                const end = result.boundary.points[1];
                const startPoint = toCanvasPoint(start.x1, start.x2, width, height, padding);
                const endPoint = toCanvasPoint(end.x1, end.x2, width, height, padding);
                ctx.moveTo(startPoint.x, startPoint.y);
                ctx.lineTo(endPoint.x, endPoint.y);
            }
            ctx.stroke();
        } else if (result.model_type === "mlp") {
            drawMlpContour(ctx, result.grid, width, height, padding);
        }

        result.points.forEach((point) => {
            const canvasPoint = toCanvasPoint(point.x1, point.x2, width, height, padding);
            ctx.beginPath();
            ctx.arc(canvasPoint.x, canvasPoint.y, 11, 0, Math.PI * 2);
            ctx.fillStyle = point.target === 1 ? "hsl(151, 66%, 42%)" : "hsl(39, 92%, 51%)";
            ctx.fill();
            ctx.lineWidth = point.target === point.prediction ? 4 : 5;
            ctx.strokeStyle = point.target === point.prediction ? "#ffffff" : "hsl(4, 78%, 56%)";
            ctx.stroke();

            ctx.fillStyle = "hsl(218, 48%, 11%)";
            ctx.font = "700 12px Inter, system-ui, sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(String(point.prediction), canvasPoint.x, canvasPoint.y);
        });
    };

    const drawMlpContour = (ctx, grid, width, height, padding) => {
        if (!Array.isArray(grid) || grid.length === 0) {
            return;
        }
        const nearBoundary = grid.filter((cell) => Math.abs(cell.score - 0.5) < 0.035);
        ctx.fillStyle = "hsl(214, 82%, 44%)";
        nearBoundary.forEach((cell) => {
            const point = toCanvasPoint(cell.x1, cell.x2, width, height, padding);
            ctx.beginPath();
            ctx.arc(point.x, point.y, 2.1, 0, Math.PI * 2);
            ctx.fill();
        });
    };

    const renderTruthTable = (points) => {
        outputs.truthTable.innerHTML = points.map((point) => {
            const pillClass = point.prediction === 1 ? "one" : "zero";
            return `
                <tr>
                    <td>${point.x1}</td>
                    <td>${point.x2}</td>
                    <td>${point.target}</td>
                    <td><span class="prediction-pill ${pillClass}">${point.prediction}</span></td>
                    <td>${Number(point.score).toFixed(3)}</td>
                </tr>
            `;
        }).join("");
    };

    const updateMetrics = (result) => {
        outputs.accuracyMetric.textContent = `${Math.round(result.accuracy * 100)}%`;
        outputs.datasetMetric.textContent = result.dataset;
        outputs.modelMetric.textContent = result.model_type === "mlp" ? "MLP" : "Perceptron";
        outputs.historyMetric.textContent = result.history.length;
        outputs.conceptText.textContent = result.concept;
        setBadge(result.accuracy === 1 ? "Solved" : "Needs tuning", result.accuracy === 1 ? "ok" : "error");

        if (result.model_type === "perceptron" && result.weights) {
            controls.w1.value = clamp(result.weights.w1, -3, 3);
            controls.w2.value = clamp(result.weights.w2, -3, 3);
            controls.bias.value = clamp(result.weights.bias, -3, 3);
            syncSliderLabels();
        }
    };

    const runModel = async () => {
        syncSliderLabels();
        setStatus("Running");
        setBadge("Running");

        const apiBase = controls.apiBase.value.replace(/\/+$/, "");
        localStorage.setItem("dlt_api_base", apiBase);

        const payload = {
            dataset: controls.dataset.value,
            model_type: controls.modelType.value,
            learning_rate: Number(controls.learningRate.value),
            epochs: Number(controls.epochs.value),
            train: controls.trainToggle.checked,
            weights: {
                w1: Number(controls.w1.value),
                w2: Number(controls.w2.value),
                bias: Number(controls.bias.value),
            },
            grid_steps: 45,
        };

        const response = await fetch(`${apiBase}/api/perceptron/train`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || "Backend request failed");
        }

        const result = await response.json();
        setStatus("Connected", "ok");
        renderTruthTable(result.points);
        updateMetrics(result);
        drawDecisionCanvas(result);
    };

    const updateModelControls = () => {
        const isMlp = controls.modelType.value === "mlp";
        outputs.weightControls.style.opacity = isMlp ? "0.48" : "1";
        outputs.weightControls.querySelectorAll("input").forEach((input) => {
            input.disabled = isMlp;
        });
        controls.trainToggle.disabled = isMlp;
    };

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        try {
            await runModel();
        } catch (error) {
            console.error(error);
            setStatus("Offline", "error");
            setBadge("Backend offline", "error");
        }
    });

    Object.values(controls).forEach((control) => {
        if (!control || control === controls.apiBase || control === controls.epochs) {
            return;
        }
        control.addEventListener("input", () => {
            syncSliderLabels();
            updateModelControls();
        });
    });

    [controls.dataset, controls.modelType, controls.trainToggle, controls.learningRate].forEach((control) => {
        control.addEventListener("change", () => form.requestSubmit());
    });

    [controls.w1, controls.w2, controls.bias].forEach((control) => {
        control.addEventListener("input", () => {
            if (!controls.trainToggle.checked && controls.modelType.value === "perceptron") {
                window.clearTimeout(control.updateTimer);
                control.updateTimer = window.setTimeout(() => form.requestSubmit(), 160);
            }
        });
    });

    controls.apiBase.addEventListener("change", () => form.requestSubmit());
    controls.epochs.addEventListener("change", () => form.requestSubmit());
    window.addEventListener("resize", () => form.requestSubmit());

    syncSliderLabels();
    updateModelControls();
    form.requestSubmit();
}

(() => {
    const apiBaseInput = document.getElementById("apiBase");
    const getApiBase = () => {
        const value = apiBaseInput?.value || localStorage.getItem("dlt_api_base") || "http://localhost:8000";
        return value.replace(/\/+$/, "");
    };

    const postJson = async (path, payload) => {
        const response = await fetch(`${getApiBase()}${path}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || "Request failed");
        }
        return response.json();
    };

    const prepareCanvas = (canvasElement) => {
        const ctx = canvasElement.getContext("2d");
        const rect = canvasElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const width = Math.max(320, Math.floor(rect.width));
        const height = Math.max(200, Math.floor(rect.height));
        canvasElement.width = width * dpr;
        canvasElement.height = height * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        return { ctx, width, height };
    };

    const drawLineChart = (canvasElement, points, xKey, yKey, options = {}) => {
        const { ctx, width, height } = prepareCanvas(canvasElement);
        const padding = { top: 24, right: 24, bottom: 34, left: 44 };
        const xValues = points.map((point) => point[xKey]);
        const yValues = points.map((point) => point[yKey]);
        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        const yMin = options.yMin ?? Math.min(...yValues);
        const yMax = options.yMax ?? Math.max(...yValues);
        const ySpan = yMax - yMin || 1;
        const xSpan = xMax - xMin || 1;
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;
        const toX = (value) => padding.left + ((value - xMin) / xSpan) * plotWidth;
        const toY = (value) => padding.top + plotHeight - ((value - yMin) / ySpan) * plotHeight;

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "hsl(210, 35%, 99%)";
        ctx.fillRect(0, 0, width, height);

        if (options.highlightVanishing) {
            points.forEach((point, index) => {
                if (point.state === "vanishing" || point.state === "dead") {
                    const x = toX(point[xKey]);
                    const next = points[index + 1] ? toX(points[index + 1][xKey]) : x + 3;
                    ctx.fillStyle = point.state === "dead" ? "hsla(4, 78%, 56%, 0.11)" : "hsla(39, 92%, 51%, 0.12)";
                    ctx.fillRect(x, padding.top, Math.max(2, next - x), plotHeight);
                }
            });
        }

        ctx.strokeStyle = "hsl(214, 30%, 88%)";
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i += 1) {
            const y = padding.top + (plotHeight / 4) * i;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(width - padding.right, y);
            ctx.stroke();
        }

        if (yMin < 0 && yMax > 0) {
            ctx.strokeStyle = "hsl(218, 24%, 30%)";
            ctx.beginPath();
            ctx.moveTo(padding.left, toY(0));
            ctx.lineTo(width - padding.right, toY(0));
            ctx.stroke();
        }

        ctx.strokeStyle = options.color || "hsl(214, 82%, 44%)";
        ctx.lineWidth = 3;
        ctx.beginPath();
        points.forEach((point, index) => {
            const x = toX(point[xKey]);
            const y = toY(point[yKey]);
            if (index === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        ctx.fillStyle = "hsl(216, 13%, 48%)";
        ctx.font = "600 12px Inter, system-ui, sans-serif";
        ctx.fillText(options.label || yKey, padding.left, 16);
    };

    const activationForm = document.getElementById("activationForm");
    if (activationForm) {
        const activationType = document.getElementById("activationType");
        const activationCanvas = document.getElementById("activationCanvas");
        const gradientCanvas = document.getElementById("gradientCanvas");
        const vanishingMetric = document.getElementById("vanishingMetric");
        const activationConcept = document.getElementById("activationConcept");
        const activationState = document.getElementById("activationState");

        const runActivation = async () => {
            activationState.textContent = "Running";
            const result = await postJson("/api/activation/curve", { function: activationType.value });
            drawLineChart(activationCanvas, result.samples, "x", "y", {
                color: "hsl(214, 82%, 44%)",
                label: `${result.function} output`,
                highlightVanishing: true,
            });
            drawLineChart(gradientCanvas, result.samples, "x", "gradient", {
                color: "hsl(151, 66%, 42%)",
                label: "gradient strength",
                yMin: -0.05,
                yMax: Math.max(1, result.max_gradient),
                highlightVanishing: true,
            });
            vanishingMetric.textContent = `${Math.round(result.vanishing_ratio * 100)}%`;
            activationConcept.textContent = result.concept;
            activationState.textContent = "Rendered";
            activationState.className = "api-status ok";
        };

        activationForm.addEventListener("change", () => runActivation().catch(() => {
            activationState.textContent = "Offline";
            activationState.className = "api-status error";
        }));
        runActivation().catch(() => {
            activationState.textContent = "Offline";
            activationState.className = "api-status error";
        });
    }

    const gdForm = document.getElementById("gdForm");
    if (gdForm) {
        const gdLearningRate = document.getElementById("gdLearningRate");
        const gdLearningRateValue = document.getElementById("gdLearningRateValue");
        const gdEpochs = document.getElementById("gdEpochs");
        const gdStart = document.getElementById("gdStart");
        const gdStartValue = document.getElementById("gdStartValue");
        const convergenceMetric = document.getElementById("convergenceMetric");
        const gdState = document.getElementById("gdState");
        const lossCanvas = document.getElementById("lossCanvas");
        const weightCanvas = document.getElementById("weightCanvas");

        const syncGdLabels = () => {
            gdLearningRateValue.value = Number(gdLearningRate.value).toFixed(3);
            gdStartValue.value = Number(gdStart.value).toFixed(2);
        };

        const runGd = async () => {
            syncGdLabels();
            gdState.textContent = "Running";
            const result = await postJson("/api/gradient-descent/simulate", {
                learning_rate: Number(gdLearningRate.value),
                epochs: Number(gdEpochs.value),
                initial_weight: Number(gdStart.value),
            });
            drawLineChart(lossCanvas, result.trace, "epoch", "loss", {
                color: "hsl(4, 78%, 56%)",
                label: "loss curve",
                yMin: 0,
            });
            drawLineChart(weightCanvas, result.trace, "epoch", "weight", {
                color: "hsl(214, 82%, 44%)",
                label: "weight path",
            });
            convergenceMetric.textContent = result.convergence;
            gdState.textContent = "Simulated";
            gdState.className = "api-status ok";
        };

        let gdTimer;
        gdForm.addEventListener("input", () => {
            syncGdLabels();
            window.clearTimeout(gdTimer);
            gdTimer = window.setTimeout(() => runGd().catch(() => {
                gdState.textContent = "Offline";
                gdState.className = "api-status error";
            }), 140);
        });
        runGd().catch(() => {
            gdState.textContent = "Offline";
            gdState.className = "api-status error";
        });
    }

    const visionForm = document.getElementById("visionForm");
    if (visionForm) {
        const visionTask = document.getElementById("visionTask");
        const visionImage = document.getElementById("visionImage");
        const visionOutput = document.getElementById("visionOutput");
        const visionEmpty = document.getElementById("visionEmpty");
        const visionPredictions = document.getElementById("visionPredictions");
        const visionState = document.getElementById("visionState");

        visionForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            if (!visionImage.files.length) {
                visionState.textContent = "Pick image";
                visionState.className = "api-status error";
                return;
            }
            visionState.textContent = "Analyzing";
            const data = new FormData();
            data.append("task", visionTask.value);
            data.append("image", visionImage.files[0]);
            try {
                const response = await fetch(`${getApiBase()}/api/vision/analyze`, {
                    method: "POST",
                    body: data,
                });
                if (!response.ok) throw new Error("Vision request failed");
                const result = await response.json();
                visionOutput.src = result.annotated_image;
                visionOutput.style.display = "block";
                visionEmpty.style.display = "none";
                visionPredictions.innerHTML = result.predictions.map((prediction) => (
                    `<span><strong>${prediction.label}</strong>${Math.round(prediction.confidence * 100)}%</span>`
                )).join("");
                visionState.textContent = `${result.boxes.length} boxes`;
                visionState.className = "api-status ok";
            } catch (error) {
                console.error(error);
                visionState.textContent = "Vision offline";
                visionState.className = "api-status error";
            }
        });
    }

    const webcamToggle = document.getElementById("webcamToggle");
    if (webcamToggle) {
        const video = document.getElementById("webcamVideo");
        const webcamCanvas = document.getElementById("webcamCanvas");
        let stream = null;
        let running = false;
        let detector = null;
        let backendBoxes = [];
        let lastBackendFrame = 0;

        const drawWebcamBoxes = (boxes = []) => {
            const { ctx, width, height } = prepareCanvas(webcamCanvas);
            ctx.clearRect(0, 0, width, height);
            if (video.readyState >= 2) {
                ctx.drawImage(video, 0, 0, width, height);
            }
            boxes.forEach((box) => {
                const x = box.x ?? box.boundingBox?.x ?? 0;
                const y = box.y ?? box.boundingBox?.y ?? 0;
                const w = box.width ?? box.boundingBox?.width ?? 0;
                const h = box.height ?? box.boundingBox?.height ?? 0;
                const scaleX = width / (video.videoWidth || width);
                const scaleY = height / (video.videoHeight || height);
                ctx.strokeStyle = "hsl(151, 66%, 42%)";
                ctx.lineWidth = 3;
                ctx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);
                ctx.fillStyle = "hsl(151, 66%, 42%)";
                ctx.fillRect(x * scaleX, Math.max(0, y * scaleY - 24), 92, 24);
                ctx.fillStyle = "#fff";
                ctx.font = "700 13px Inter, system-ui, sans-serif";
                ctx.fillText("face", x * scaleX + 8, Math.max(17, y * scaleY - 7));
            });
        };

        const analyzeWebcamFrame = async () => {
            if (video.readyState < 2) return;
            const scratch = document.createElement("canvas");
            scratch.width = video.videoWidth || 640;
            scratch.height = video.videoHeight || 360;
            scratch.getContext("2d").drawImage(video, 0, 0, scratch.width, scratch.height);
            const blob = await new Promise((resolve) => scratch.toBlob(resolve, "image/jpeg", 0.72));
            if (!blob) return;
            const data = new FormData();
            data.append("task", "face_detection");
            data.append("image", blob, "webcam-frame.jpg");
            const response = await fetch(`${getApiBase()}/api/vision/analyze`, {
                method: "POST",
                body: data,
            });
            if (response.ok) {
                const result = await response.json();
                backendBoxes = result.boxes || [];
            }
        };

        const loop = async () => {
            if (!running) return;
            let boxes = [];
            if (detector && video.readyState >= 2) {
                try {
                    boxes = await detector.detect(video);
                } catch (error) {
                    boxes = [];
                }
            } else if (performance.now() - lastBackendFrame > 900) {
                lastBackendFrame = performance.now();
                analyzeWebcamFrame().catch(() => {
                    backendBoxes = [];
                });
                boxes = backendBoxes;
            } else {
                boxes = backendBoxes;
            }
            drawWebcamBoxes(boxes);
            requestAnimationFrame(loop);
        };

        webcamToggle.addEventListener("click", async () => {
            if (running) {
                running = false;
                stream?.getTracks().forEach((track) => track.stop());
                stream = null;
                backendBoxes = [];
                webcamToggle.textContent = "Open webcam";
                drawWebcamBoxes([]);
                return;
            }
            stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            video.srcObject = stream;
            await video.play();
            if ("FaceDetector" in window) {
                detector = new window.FaceDetector({ fastMode: true, maxDetectedFaces: 8 });
            }
            running = true;
            webcamToggle.textContent = "Stop webcam";
            loop();
        });
    }
})();
