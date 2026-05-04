window.DLT = window.DLT || {};

DLT.visionView = (() => {
    const initImageUpload = () => {
        const visionForm = document.getElementById("visionForm");
        if (!visionForm) return;

        visionForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            const visionTask = document.getElementById("visionTask");
            const visionImage = document.getElementById("visionImage");
            const visionOutput = document.getElementById("visionOutput");
            const visionEmpty = document.getElementById("visionEmpty");
            const visionPredictions = document.getElementById("visionPredictions");
            const visionState = document.getElementById("visionState");

            if (!visionImage.files.length) {
                visionState.textContent = "Pick image";
                visionState.className = "api-status error";
                return;
            }

            visionState.textContent = "Analyzing";
            const data = new FormData();
            data.append("task", visionTask.value);
            data.append("image", visionImage.files[0]);

            try {
                const result = await DLT.api.postForm("/api/vision/analyze", data);
                visionOutput.src = result.annotated_image;
                visionOutput.style.display = "block";
                visionEmpty.style.display = "none";
                visionPredictions.innerHTML = result.predictions.map((prediction) => (
                    `<span><strong>${prediction.label}</strong>${Math.round(prediction.confidence * 100)}%</span>`
                )).join("");
                visionState.textContent = `${result.boxes.length} boxes`;
                visionState.className = "api-status ok";
            } catch (error) {
                console.error(error);
                visionState.textContent = "Vision offline";
                visionState.className = "api-status error";
            }
        });
    };

    const initWebcam = () => {
        const webcamToggle = document.getElementById("webcamToggle");
        if (!webcamToggle) return;

        const video = document.getElementById("webcamVideo");
        const webcamCanvas = document.getElementById("webcamCanvas");
        let stream = null;
        let running = false;
        let detector = null;
        let backendBoxes = [];
        let lastBackendFrame = 0;

        const drawWebcamBoxes = (boxes = []) => {
            const { ctx, width, height } = DLT.charts.prepareCanvas(webcamCanvas);
            ctx.clearRect(0, 0, width, height);
            if (video.readyState >= 2) {
                ctx.drawImage(video, 0, 0, width, height);
            }
            boxes.forEach((box) => {
                const x = box.x ?? box.boundingBox?.x ?? 0;
                const y = box.y ?? box.boundingBox?.y ?? 0;
                const w = box.width ?? box.boundingBox?.width ?? 0;
                const h = box.height ?? box.boundingBox?.height ?? 0;
                const scaleX = width / (video.videoWidth || width);
                const scaleY = height / (video.videoHeight || height);
                ctx.strokeStyle = "hsl(151, 66%, 42%)";
                ctx.lineWidth = 3;
                ctx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);
                ctx.fillStyle = "hsl(151, 66%, 42%)";
                ctx.fillRect(x * scaleX, Math.max(0, y * scaleY - 24), 92, 24);
                ctx.fillStyle = "#fff";
                ctx.font = "700 13px Inter, system-ui, sans-serif";
                ctx.fillText("face", x * scaleX + 8, Math.max(17, y * scaleY - 7));
            });
        };

        const analyzeWebcamFrame = async () => {
            if (video.readyState < 2) return;
            const scratch = document.createElement("canvas");
            scratch.width = video.videoWidth || 640;
            scratch.height = video.videoHeight || 360;
            scratch.getContext("2d").drawImage(video, 0, 0, scratch.width, scratch.height);
            const blob = await new Promise((resolve) => scratch.toBlob(resolve, "image/jpeg", 0.72));
            if (!blob) return;
            const data = new FormData();
            data.append("task", "face_detection");
            data.append("image", blob, "webcam-frame.jpg");
            const result = await DLT.api.postForm("/api/vision/analyze", data);
            backendBoxes = result.boxes || [];
        };

        const loop = async () => {
            if (!running) return;
            let boxes = [];
            if (detector && video.readyState >= 2) {
                try {
                    boxes = await detector.detect(video);
                } catch (error) {
                    boxes = [];
                }
            } else if (performance.now() - lastBackendFrame > 900) {
                lastBackendFrame = performance.now();
                analyzeWebcamFrame().catch(() => {
                    backendBoxes = [];
                });
                boxes = backendBoxes;
            } else {
                boxes = backendBoxes;
            }
            drawWebcamBoxes(boxes);
            requestAnimationFrame(loop);
        };

        webcamToggle.addEventListener("click", async () => {
            if (running) {
                running = false;
                stream?.getTracks().forEach((track) => track.stop());
                stream = null;
                backendBoxes = [];
                webcamToggle.textContent = "Open webcam";
                drawWebcamBoxes([]);
                return;
            }
            stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            video.srcObject = stream;
            await video.play();
            if ("FaceDetector" in window) {
                detector = new window.FaceDetector({ fastMode: true, maxDetectedFaces: 8 });
            }
            running = true;
            webcamToggle.textContent = "Stop webcam";
            loop();
        });
    };

    const init = () => {
        initImageUpload();
        initWebcam();
    };

    return { init };
})();
