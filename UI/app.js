"use strict";

const samples = {
    kohli: { name: "Virat Kohli", url: "/assets/test_images/kohli_1.jpg" },
    serena: { name: "Serena Williams", url: "/samples/serena.jpg" },
    pair: { name: "Two faces", url: "/assets/test_images/sharapova_3.jpg" },
};

const byId = (id) => document.getElementById(id);
const form = byId("upload-form");
const input = byId("photo-input");
const dropArea = byId("drop-area");
const preview = byId("photo-preview");
const submit = byId("classify");
const status = byId("upload-status");
const error = byId("upload-error");

let image = null;
let previewUrl = null;
let operation = null;

function showMessage(element, message) {
    element.textContent = message;
}

function resetResults(message = "Choose a photo to see predictions.") {
    byId("results").hidden = true;
    byId("empty-results").hidden = false;
    byId("empty-message").textContent = message;
    byId("face-boxes").replaceChildren();
}

function clearPhoto() {
    operation?.abort();
    operation = null;
    image = null;
    input.value = "";
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    preview.removeAttribute("src");
    byId("preview-frame").hidden = true;
    byId("photo-actions").hidden = true;
    byId("upload-prompt").hidden = false;
    byId("file-label").textContent = "Choose photo";
    submit.disabled = true;
    showMessage(status, "");
    showMessage(error, "");
    resetResults();
}

function showPhoto(file, name) {
    image = file;
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    byId("preview-frame").hidden = false;
    byId("photo-actions").hidden = false;
    byId("upload-prompt").hidden = true;
    byId("file-label").textContent = "Change photo";
    byId("photo-name").textContent = name;
    submit.disabled = false;
    resetResults("Classify the selected photo to see predictions.");
}

function choosePhoto(files) {
    if (!files.length) return;
    clearPhoto();
    if (files.length !== 1) {
        showMessage(error, "Choose one photo at a time.");
        return;
    }
    const [file] = files;
    if (!/\.(jpe?g|png|webp)$/i.test(file.name) && !/^image\/(jpeg|png|webp)$/.test(file.type)) {
        showMessage(error, "Choose a JPEG, PNG or WebP photo.");
        return;
    }
    if (file.size > 8 * 1024 * 1024) {
        showMessage(error, "Choose a photo that is 8 MiB or smaller.");
        return;
    }
    showPhoto(file, file.name);
}

function displayName(label) {
    return label.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function displayScore(score) {
    return `${score.toFixed(2)}%`;
}

function scoreRow({ label, score }) {
    const row = document.createElement("li");
    const name = document.createElement("span");
    const value = document.createElement("span");
    const track = document.createElement("span");
    const fill = document.createElement("span");
    row.className = "score-row";
    name.textContent = displayName(label);
    value.textContent = displayScore(score);
    track.className = "score-track";
    track.setAttribute("aria-hidden", "true");
    fill.className = "score-fill";
    fill.style.width = displayScore(score);
    track.append(fill);
    row.append(name, value, track);
    return row;
}

function drawBox({ box }, width, height, index) {
    const frame = document.createElement("div");
    const number = document.createElement("span");
    frame.className = "face-box";
    frame.style.left = `${100 * box.x / width}%`;
    frame.style.top = `${100 * box.y / height}%`;
    frame.style.width = `${100 * box.width / width}%`;
    frame.style.height = `${100 * box.height / height}%`;
    number.textContent = index + 1;
    frame.append(number);
    return frame;
}

function renderResults(result) {
    const { faces, detected_faces: detected, image: { width, height } } = result;
    if (!faces.length) {
        resetResults("No face with two visible eyes was detected. Try a clear, front-facing photo.");
        showMessage(status, "No qualifying face detected.");
        return;
    }

    const picker = byId("face-picker");
    const boxes = faces.map((face, index) => drawBox(face, width, height, index));
    const buttons = faces.map((_, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = `Face ${index + 1}`;
        button.addEventListener("click", () => selectFace(index));
        return button;
    });

    function selectFace(index) {
        boxes.forEach((box, position) => box.classList.toggle("selected", position === index));
        buttons.forEach((button, position) => button.setAttribute("aria-pressed", position === index));
        const { label, scores } = faces[index];
        byId("prediction-name").textContent = displayName(label);
        byId("prediction-score").textContent = displayScore(
            scores.find((entry) => entry.label === label).score,
        );
        byId("score-list").replaceChildren(...scores.map(scoreRow));
    }

    byId("face-boxes").replaceChildren(...boxes);
    picker.replaceChildren(...buttons);
    picker.hidden = faces.length < 2;
    byId("empty-results").hidden = true;
    byId("results").hidden = false;
    selectFace(0);
    const message = detected > faces.length
        ? `Showing ${faces.length} of ${detected} detected faces.`
        : `${detected} ${detected === 1 ? "face" : "faces"} detected.`;
    showMessage(status, message);
}

async function classify(file, signal) {
    submit.disabled = true;
    showMessage(status, "Analyzing photo…");
    showMessage(error, "");
    resetResults("Analyzing photo…");
    try {
        const data = new FormData();
        data.append("image", file);
        const response = await fetch("/api/v1/classify", { method: "POST", body: data, signal });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error?.message || "The photo could not be classified.");
        }
        if (!signal.aborted) renderResults(result);
    } catch (failure) {
        if (!signal.aborted) {
            showMessage(status, "");
            showMessage(error, failure instanceof TypeError || failure instanceof SyntaxError
                ? "Could not reach the classifier. Try again."
                : failure.message);
            resetResults("Try the photo again or choose another one.");
        }
    } finally {
        if (!signal.aborted) submit.disabled = false;
    }
}

form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!image) return;
    operation?.abort();
    operation = new AbortController();
    void classify(image, operation.signal);
});

input.addEventListener("change", () => choosePhoto(Array.from(input.files)));
byId("reset-photo").addEventListener("click", () => {
    clearPhoto();
    input.focus();
});

dropArea.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropArea.classList.add("is-dragging");
});
dropArea.addEventListener("dragleave", (event) => {
    if (!dropArea.contains(event.relatedTarget)) dropArea.classList.remove("is-dragging");
});
dropArea.addEventListener("drop", (event) => {
    event.preventDefault();
    dropArea.classList.remove("is-dragging");
    choosePhoto(Array.from(event.dataTransfer.files));
});

document.querySelectorAll("[data-sample]").forEach((button) => {
    button.addEventListener("click", async () => {
        clearPhoto();
        operation = new AbortController();
        const { signal } = operation;
        const { name, url } = samples[button.dataset.sample];
        showMessage(status, "Loading sample…");
        try {
            const response = await fetch(url, { signal });
            if (!response.ok) throw new Error("Sample unavailable.");
            const blob = await response.blob();
            if (signal.aborted) return;
            showPhoto(new File([blob], url.split("/").at(-1), { type: blob.type }), name);
            await classify(image, signal);
        } catch {
            if (!signal.aborted) {
                showMessage(status, "");
                showMessage(error, "Could not load the sample. Try again.");
            }
        }
    });
});
