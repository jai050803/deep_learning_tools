window.DLT = window.DLT || {};

DLT.rnnView = (() => {
    const render = (result) => {
        document.getElementById("rnnPrediction").textContent = result.prediction;
        document.getElementById("rnnTokens").textContent = `tokens: ${result.tokens.join(" / ")}`;
        document.getElementById("rnnConcept").textContent = result.concept;
        document.getElementById("rnnPredictions").innerHTML = result.predictions.map((item) => {
            const percent = Math.round(item.confidence * 100);
            return `
                <div class="prediction-bar">
                    <span>${item.word}</span>
                    <div><i style="width:${percent}%"></i></div>
                    <strong>${percent}%</strong>
                </div>
            `;
        }).join("");
        document.getElementById("rnnHiddenState").innerHTML = result.hidden_state_preview.map((value, index) => (
            `<span><small>h${index + 1}</small>${value}</span>`
        )).join("");
    };

    const run = async () => {
        const rnnState = document.getElementById("rnnState");
        rnnState.textContent = "Predicting";
        const result = await DLT.api.postJson("/api/rnn/predict", {
            text: document.getElementById("rnnText").value,
            top_k: 3,
        });
        render(result);
        rnnState.textContent = "Predicted";
        rnnState.className = "api-status ok";
    };

    const init = () => {
        const rnnForm = document.getElementById("rnnForm");
        if (!rnnForm) return;
        rnnForm.addEventListener("submit", (event) => {
            event.preventDefault();
            run().catch((error) => {
                console.error(error);
                const rnnState = document.getElementById("rnnState");
                rnnState.textContent = "Offline";
                rnnState.className = "api-status error";
            });
        });
        document.addEventListener("dlt:api-base-change", () => run().catch(() => {}));
        run().catch(() => {
            const rnnState = document.getElementById("rnnState");
            rnnState.textContent = "Offline";
            rnnState.className = "api-status error";
        });
    };

    return { init };
})();
