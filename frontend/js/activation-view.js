window.DLT = window.DLT || {};

DLT.activationView = (() => {
    let lastResult = null;

    const run = async () => {
        const activationType = document.getElementById("activationType");
        const activationCanvas = document.getElementById("activationCanvas");
        const gradientCanvas = document.getElementById("gradientCanvas");
        const vanishingMetric = document.getElementById("vanishingMetric");
        const activationConcept = document.getElementById("activationConcept");
        const activationState = document.getElementById("activationState");

        activationState.textContent = "Running";
        const result = await DLT.api.postJson("/api/activation/curve", { function: activationType.value });
        lastResult = result;
        DLT.charts.drawLineChart(activationCanvas, result.samples, "x", "y", {
            color: "hsl(214, 82%, 44%)",
            label: `${result.function} output`,
            highlightVanishing: true,
        });
        DLT.charts.drawLineChart(gradientCanvas, result.samples, "x", "gradient", {
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

    const redraw = () => {
        if (!lastResult) return;
        const activationCanvas = document.getElementById("activationCanvas");
        const gradientCanvas = document.getElementById("gradientCanvas");
        DLT.charts.drawLineChart(activationCanvas, lastResult.samples, "x", "y", {
            color: "hsl(214, 82%, 44%)",
            label: `${lastResult.function} output`,
            highlightVanishing: true,
        });
        DLT.charts.drawLineChart(gradientCanvas, lastResult.samples, "x", "gradient", {
            color: "hsl(151, 66%, 42%)",
            label: "gradient strength",
            yMin: -0.05,
            yMax: Math.max(1, lastResult.max_gradient),
            highlightVanishing: true,
        });
    };

    const init = () => {
        const activationForm = document.getElementById("activationForm");
        if (!activationForm) return;
        activationForm.addEventListener("change", () => run().catch(() => {
            const activationState = document.getElementById("activationState");
            activationState.textContent = "Offline";
            activationState.className = "api-status error";
        }));
        document.addEventListener("dlt:api-base-change", () => run().catch(() => {}));
        run().catch(() => {
            const activationState = document.getElementById("activationState");
            activationState.textContent = "Offline";
            activationState.className = "api-status error";
        });
    };

    return { init, redraw };
})();
