window.DLT = window.DLT || {};

DLT.explainView = (() => {
    const percent = (value) => `${Math.round(value * 100)}%`;

    const syncLabels = () => {
        document.getElementById("explainLearningRateValue").value = Number(document.getElementById("explainLearningRate").value).toFixed(3);
        document.getElementById("explainEpochsValue").value = document.getElementById("explainEpochs").value;
        document.getElementById("explainNeuronsValue").value = document.getElementById("explainNeurons").value;
    };

    const render = (result) => {
        document.getElementById("healthScore").textContent = `health: ${percent(result.health_score)}`;
        document.getElementById("explainTrainAccuracy").textContent = percent(result.train_accuracy);
        document.getElementById("explainValidationAccuracy").textContent = percent(result.validation_accuracy);
        document.getElementById("explainGap").textContent = percent(result.generalization_gap);
        document.getElementById("insightList").innerHTML = result.insights.map((insight) => `
            <article class="insight-item ${insight.type}">
                <strong>${insight.title}</strong>
                <p>${insight.message}</p>
            </article>
        `).join("");
        document.getElementById("recommendationList").innerHTML = result.recommendations.map((item, index) => (
            `<span><small>${index + 1}</small>${item}</span>`
        )).join("");
    };

    const run = async () => {
        const explainState = document.getElementById("explainState");
        syncLabels();
        explainState.textContent = "Analyzing";
        const result = await DLT.api.postJson("/api/insights/explain", {
            dataset: document.getElementById("explainDataset").value,
            model_type: document.getElementById("explainModelType").value,
            learning_rate: Number(document.getElementById("explainLearningRate").value),
            epochs: Number(document.getElementById("explainEpochs").value),
            hidden_neurons: Number(document.getElementById("explainNeurons").value),
        });
        render(result);
        explainState.textContent = "Explained";
        explainState.className = "api-status ok";
    };

    const init = () => {
        const form = document.getElementById("explainForm");
        if (!form) return;
        let timer;
        form.addEventListener("submit", (event) => {
            event.preventDefault();
            run().catch((error) => {
                console.error(error);
                const explainState = document.getElementById("explainState");
                explainState.textContent = "Offline";
                explainState.className = "api-status error";
            });
        });
        form.addEventListener("input", () => {
            syncLabels();
            window.clearTimeout(timer);
            timer = window.setTimeout(() => form.requestSubmit(), 180);
        });
        form.addEventListener("change", () => form.requestSubmit());
        document.addEventListener("dlt:api-base-change", () => form.requestSubmit());
        syncLabels();
        form.requestSubmit();
    };

    return { init };
})();
