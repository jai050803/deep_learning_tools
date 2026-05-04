window.DLT = window.DLT || {};

DLT.navigation = (() => {
    const activateView = (viewId) => {
        document.querySelectorAll(".tab-item").forEach((tab) => {
            tab.classList.toggle("active", tab.dataset.viewTarget === viewId);
        });
        document.querySelectorAll(".view-panel").forEach((panel) => {
            panel.classList.toggle("active", panel.id === viewId);
        });
        window.dispatchEvent(new CustomEvent("dlt:view-change", { detail: { viewId } }));
    };

    const init = () => {
        document.querySelectorAll(".tab-item").forEach((tab) => {
            tab.addEventListener("click", () => activateView(tab.dataset.viewTarget));
        });
    };

    return { activateView, init };
})();
