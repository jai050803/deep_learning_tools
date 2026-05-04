document.addEventListener("DOMContentLoaded", () => {
    DLT.api.init();
    DLT.navigation.init();
    DLT.perceptronView.init();
    DLT.activationView.init();
    DLT.gradientView.init();
    DLT.explainView.init();
    DLT.whatIfView.init();
    DLT.visionView.init();
    DLT.rnnView.init();

    window.addEventListener("dlt:view-change", (event) => {
        const viewId = event.detail.viewId;
        if (viewId === "perceptronView") DLT.perceptronView.redraw();
        if (viewId === "activationView") DLT.activationView.redraw();
        if (viewId === "gradientView") DLT.gradientView.redraw();
        if (viewId === "whatIfView") DLT.whatIfView.redraw();
    });
});
