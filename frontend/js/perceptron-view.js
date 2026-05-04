window.DLT = window.DLT || {};

DLT.perceptronView = (() => {
    let lastResult = null;

    const form = () => document.getElementById("controlForm");
    const canvas = () => document.getElementById("decisionCanvas");
    const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
    const graphMin = -0.25;
    const graphMax = 1.25;
    const graphSpan = graphMax - graphMin;

    const getControls = () => ({
        dataset: document.getElementById("dataset"),
        modelType: document.getElementById("modelType"),
        trainToggle: document.getElementById("trainToggle"),
        w1: document.getElementById("w1"),
        w2: document.getElementById("w2"),
        bias: document.getElementById("bias"),
        learningRate: document.getElementById("learningRate"),
        epochs: document.getElementById("epochs"),
    });

    const getOutputs = () => ({
        w1: document.getElementById("w1Value"),
        w2: document.getElementById("w2Value"),
        bias: document.getElementById("biasValue"),
        learningRate: document.getElementById("learningRateValue"),
        conceptText: document.getElementById("conceptText"),
        accuracyMetric: document.getElementById("accuracyMetric"),
        datasetMetric: document.getElementById("datasetMetric"),
        modelMetric: document.getElementById("modelMetric"),
        historyMetric: document.getElementById("historyMetric"),
        resultBadge: document.getElementById("resultBadge"),
        truthTable: document.getElementById("truthTable"),
        weightControls: document.getElementById("weightControls"),
    });

    const syncSliderLabels = () => {
        const controls = getControls();
        const outputs = getOutputs();
        outputs.w1.value = Number(controls.w1.value).toFixed(2);
        outputs.w2.value = Number(controls.w2.value).toFixed(2);
        outputs.bias.value = Number(controls.bias.value).toFixed(2);
        outputs.learningRate.value = Number(controls.learningRate.value).toFixed(2);
    };

    const setBadge = (message, type = "") => {
        const badge = getOutputs().resultBadge;
        badge.textContent = message;
        badge.className = `result-badge ${type}`.trim();
    };

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
    };

    const drawMlpContour = (ctx, grid, width, height, padding) => {
        if (!Array.isArray(grid) || grid.length === 0) return;
        const nearBoundary = grid.filter((cell) => Math.abs(cell.score - 0.5) < 0.035);
        ctx.fillStyle = "hsl(214, 82%, 44%)";
        nearBoundary.forEach((cell) => {
            const point = toCanvasPoint(cell.x1, cell.x2, width, height, padding);
            ctx.beginPath();
            ctx.arc(point.x, point.y, 2.1, 0, Math.PI * 2);
            ctx.fill();
        });
    };

    const drawDecisionCanvas = (result) => {
        const target = canvas();
        const { ctx, width, height } = DLT.charts.prepareCanvas(target);
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

    const renderTruthTable = (points) => {
        getOutputs().truthTable.innerHTML = points.map((point) => {
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
        const controls = getControls();
        const outputs = getOutputs();
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

    const updateModelControls = () => {
        const controls = getControls();
        const weightControls = getOutputs().weightControls;
        const isMlp = controls.modelType.value === "mlp";
        weightControls.style.opacity = isMlp ? "0.48" : "1";
        weightControls.querySelectorAll("input").forEach((input) => {
            input.disabled = isMlp;
        });
        controls.trainToggle.disabled = isMlp;
    };

    const run = async () => {
        const controls = getControls();
        syncSliderLabels();
        DLT.api.setStatus("Running");
        setBadge("Running");
        DLT.api.saveApiBase();

        const result = await DLT.api.postJson("/api/perceptron/train", {
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
        });

        lastResult = result;
        DLT.api.setStatus("Connected", "ok");
        renderTruthTable(result.points);
        updateMetrics(result);
        drawDecisionCanvas(result);
    };

    const init = () => {
        if (!form() || !canvas()) return;
        const controls = getControls();

        form().addEventListener("submit", async (event) => {
            event.preventDefault();
            run().catch((error) => {
                console.error(error);
                DLT.api.setStatus("Offline", "error");
                setBadge("Backend offline", "error");
            });
        });

        Object.values(controls).forEach((control) => {
            control?.addEventListener("input", () => {
                syncSliderLabels();
                updateModelControls();
            });
        });

        [controls.dataset, controls.modelType, controls.trainToggle, controls.learningRate].forEach((control) => {
            control.addEventListener("change", () => form().requestSubmit());
        });

        [controls.w1, controls.w2, controls.bias].forEach((control) => {
            control.addEventListener("input", () => {
                if (!controls.trainToggle.checked && controls.modelType.value === "perceptron") {
                    window.clearTimeout(control.updateTimer);
                    control.updateTimer = window.setTimeout(() => form().requestSubmit(), 160);
                }
            });
        });

        controls.epochs.addEventListener("change", () => form().requestSubmit());
        document.addEventListener("dlt:api-base-change", () => form().requestSubmit());
        window.addEventListener("resize", () => {
            if (lastResult) drawDecisionCanvas(lastResult);
        });

        syncSliderLabels();
        updateModelControls();
        form().requestSubmit();
    };

    return { init, redraw: () => lastResult && drawDecisionCanvas(lastResult) };
})();
