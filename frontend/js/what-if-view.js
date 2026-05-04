window.DLT = window.DLT || {};

DLT.whatIfView = (() => {
    let lastResult = null;
    const percent = (value) => `${Math.round(value * 100)}%`;

    const syncLabels = () => {
        document.getElementById("whatIfNeuronsValue").value = document.getElementById("whatIfNeurons").value;
        document.getElementById("whatIfLearningRateValue").value = Number(document.getElementById("whatIfLearningRate").value).toFixed(3);
        document.getElementById("whatIfEpochsValue").value = document.getElementById("whatIfEpochs").value;
        document.getElementById("whatIfNoiseValue").value = Number(document.getElementById("whatIfNoise").value).toFixed(2);
    };

    const draw = (result) => {
        DLT.charts.drawLineChart(document.getElementById("whatIfLossCanvas"), result.loss_trace, "epoch", "loss", {
            color: result.behavior === "unstable" ? "hsl(4, 78%, 56%)" : "hsl(214, 82%, 44%)",
            label: "simulated loss",
            yMin: 0,
        });
    };

    const render = (result) => {
        document.getElementById("whatIfTrainAccuracy").textContent = percent(result.train_accuracy);
        document.getElementById("whatIfValidationAccuracy").textContent = percent(result.validation_accuracy);
        document.getElementById("whatIfGap").textContent = percent(result.generalization_gap);
        document.getElementById("whatIfBehavior").textContent = result.behavior;
        draw(result);
    };

    const run = async () => {
        const whatIfState = document.getElementById("whatIfState");
        syncLabels();
        whatIfState.textContent = "Simulating";
        const result = await DLT.api.postJson("/api/insights/what-if", {
            dataset: document.getElementById("whatIfDataset").value,
            hidden_neurons: Number(document.getElementById("whatIfNeurons").value),
            learning_rate: Number(document.getElementById("whatIfLearningRate").value),
            epochs: Number(document.getElementById("whatIfEpochs").value),
            noise: Number(document.getElementById("whatIfNoise").value),
        });
        lastResult = result;
        render(result);
        whatIfState.textContent = "Live";
        whatIfState.className = "api-status ok";
    };

    const init = () => {
        const form = document.getElementById("whatIfForm");
        if (!form) return;
        let timer;
        form.addEventListener("input", () => {
            syncLabels();
            window.clearTimeout(timer);
            timer = window.setTimeout(() => run().catch((error) => {
                console.error(error);
                const whatIfState = document.getElementById("whatIfState");
                whatIfState.textContent = "Offline";
                whatIfState.className = "api-status error";
            }), 120);
        });
        form.addEventListener("change", () => run().catch(() => {}));
        document.addEventListener("dlt:api-base-change", () => run().catch(() => {}));
        syncLabels();
        run().catch(() => {
            const whatIfState = document.getElementById("whatIfState");
            whatIfState.textContent = "Offline";
            whatIfState.className = "api-status error";
        });
    };

    return { init, redraw: () => lastResult && draw(lastResult) };
})();
