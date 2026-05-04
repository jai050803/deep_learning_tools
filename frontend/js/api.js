window.DLT = window.DLT || {};

DLT.api = (() => {
    const apiBaseInput = () => document.getElementById("apiBase");
    const apiStatus = () => document.getElementById("apiStatus");

    const getApiBase = () => {
        const value = apiBaseInput()?.value || localStorage.getItem("dlt_api_base") || "https://deep-learning-toolbox-api.onrender.com";
        return value.replace(/\/+$/, "");
    };

    const saveApiBase = () => {
        const base = getApiBase();
        localStorage.setItem("dlt_api_base", base);
        if (apiBaseInput()) {
            apiBaseInput().value = base;
        }
        return base;
    };

    const setStatus = (message, type = "") => {
        const status = apiStatus();
        if (!status) return;
        status.textContent = message;
        status.className = `api-status ${type}`.trim();
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

    const postForm = async (path, data) => {
        const response = await fetch(`${getApiBase()}${path}`, {
            method: "POST",
            body: data,
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || "Request failed");
        }
        return response.json();
    };

    const init = () => {
        const storedApiBase = localStorage.getItem("dlt_api_base");
        if (storedApiBase && apiBaseInput()) {
            apiBaseInput().value = storedApiBase;
        }
        apiBaseInput()?.addEventListener("change", () => {
            saveApiBase();
            document.dispatchEvent(new CustomEvent("dlt:api-base-change"));
        });
    };

    return { getApiBase, init, postForm, postJson, saveApiBase, setStatus };
})();
