window.DLT = window.DLT || {};

DLT.charts = (() => {
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

    return { drawLineChart, prepareCanvas };
})();
