window.DLT = window.DLT || {};

DLT.gradientView = (() => {
    let lastResult = null;

    const syncLabels = () => {
        document.getElementById("gdLearningRateValue").value = Number(document.getElementById("gdLearningRate").value).toFixed(3);
        document.getElementById("gdStartValue").value = Number(document.getElementById("gdStart").value).toFixed(2);
    };

    const draw = (result) => {
        DLT.charts.drawLineChart(document.getElementById("lossCanvas"), result.trace, "epoch", "loss", {
            color: "hsl(4, 78%, 56%)",
            label: "loss curve",
            yMin: 0,
        });
        DLT.charts.drawLineChart(document.getElementById("weightCanvas"), result.trace, "epoch", "weight", {
            color: "hsl(214, 82%, 44%)",
            label: "weight path",
        });
    };

    const run = async () => {
        const gdState = document.getElementById("gdState");
        syncLabels();
        gdState.textContent = "Running";
        const result = await DLT.api.postJson("/api/gradient-descent/simulate", {
            learning_rate: Number(document.getElementById("gdLearningRate").value),
            epochs: Number(document.getElementById("gdEpochs").value),
            initial_weight: Number(document.getElementById("gdStart").value),
        });
        lastResult = result;
        draw(result);
        document.getElementById("convergenceMetric").textContent = result.convergence;
        gdState.textContent = "Simulated";
        gdState.className = "api-status ok";
    };

    const init = () => {
        const gdForm = document.getElementById("gdForm");
        if (!gdForm) return;
        let timer;
        gdForm.addEventListener("input", () => {
            syncLabels();
            window.clearTimeout(timer);
            timer = window.setTimeout(() => run().catch(() => {
                const gdState = document.getElementById("gdState");
                gdState.textContent = "Offline";
                gdState.className = "api-status error";
            }), 140);
        });
        document.addEventListener("dlt:api-base-change", () => run().catch(() => {}));
        run().catch(() => {
            const gdState = document.getElementById("gdState");
            gdState.textContent = "Offline";
            gdState.className = "api-status error";
        });
    };

    return { init, redraw: () => lastResult && draw(lastResult) };
})();
