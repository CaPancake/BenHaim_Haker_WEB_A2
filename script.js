const classNames = ["cat", "dog", "horse"];
const MODEL_URL = "animal_cnn.onnx";
const MODEL_KEY = "animal_cnn_onnx_debug_v1";

// Helps ONNX Runtime Web find its WebAssembly backend files on GitHub Pages
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

let session = null;

async function arrayBufferToBase64(buffer) {
  let binary = "";
  const bytes = new Uint8Array(buffer);

  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }

  return btoa(binary);
}

function base64ToUint8Array(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);

  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }

  return bytes;
}

function inspectModelBytes(buffer) {
  const bytes = new Uint8Array(buffer);
  const firstBytes = bytes.slice(0, 200);
  const firstText = new TextDecoder("utf-8", { fatal: false }).decode(firstBytes);

  console.log("===== MODEL FILE DEBUG =====");
  console.log("Fetched size:", buffer.byteLength);
  console.log("First bytes as text:", firstText);
  console.log("First 20 raw bytes:", Array.from(bytes.slice(0, 20)));
  console.log("============================");

  if (firstText.startsWith("version https://git-lfs.github.com/spec")) {
    throw new Error(
      "GitHub Pages is serving a Git LFS pointer instead of the real ONNX model."
    );
  }

  if (
    firstText.trim().startsWith("<!DOCTYPE html") ||
    firstText.trim().startsWith("<html")
  ) {
    throw new Error(
      "GitHub Pages is serving HTML instead of animal_cnn.onnx. Check the model path."
    );
  }

  if (buffer.byteLength < 10000) {
    console.warn(
      "Warning: ONNX file looks very small. It may be incomplete or not the real model."
    );
  }
}

async function fetchModelBuffer() {
  const response = await fetch(MODEL_URL);

  console.log("Model fetch status:", response.status);
  console.log("Model content-type:", response.headers.get("content-type"));

  if (!response.ok) {
    throw new Error(`Could not fetch model file. Status: ${response.status}`);
  }

  const buffer = await response.arrayBuffer();
  inspectModelBytes(buffer);

  return buffer;
}

async function loadModel() {
  const status = document.getElementById("modelStatus");

  status.textContent = "Checking model file...";

  let modelBase64 = localStorage.getItem(MODEL_KEY);
  let modelBytes = null;

  if (!modelBase64) {
    status.textContent =
      "Model not found in localStorage. Loading from GitHub Pages...";

    const buffer = await fetchModelBuffer();

    modelBase64 = await arrayBufferToBase64(buffer);
    localStorage.setItem(MODEL_KEY, modelBase64);

    modelBytes = new Uint8Array(buffer);

    status.textContent = "Model saved to localStorage.";
  } else {
    status.textContent = "Model loaded from localStorage.";

    modelBytes = base64ToUint8Array(modelBase64);

    console.log("===== LOCALSTORAGE MODEL DEBUG =====");
    console.log("Model bytes from localStorage:", modelBytes.length);

    const firstText = new TextDecoder("utf-8", { fatal: false }).decode(
      modelBytes.slice(0, 200)
    );

    console.log("First localStorage bytes as text:", firstText);
    console.log("====================================");

    if (firstText.startsWith("version https://git-lfs.github.com/spec")) {
      localStorage.removeItem(MODEL_KEY);
      throw new Error(
        "localStorage contains a Git LFS pointer instead of the real model. Cleared cached model. Refresh the page."
      );
    }

    if (
      firstText.trim().startsWith("<!DOCTYPE html") ||
      firstText.trim().startsWith("<html")
    ) {
      localStorage.removeItem(MODEL_KEY);
      throw new Error(
        "localStorage contains HTML instead of the model. Cleared cached model. Refresh the page."
      );
    }
  }

  try {
    console.log("Creating ONNX Runtime session...");

    session = await ort.InferenceSession.create(modelBytes);

    console.log("Model loaded successfully!");
    console.log("Input names:", session.inputNames);
    console.log("Output names:", session.outputNames);

    status.textContent += " Ready!";
  } catch (error) {
    console.error("ONNX session creation failed:", error);
    throw error;
  }
}

function previewImage(file) {
  const preview = document.getElementById("preview");

  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
}

function preprocessImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    const mean = [0.4914, 0.4822, 0.4465];
    const std = [0.2470, 0.2435, 0.2616];

    img.onload = () => {
      try {
        const size = Math.min(img.width, img.height);
        const sx = (img.width - size) / 2;
        const sy = (img.height - size) / 2;

        ctx.clearRect(0, 0, 32, 32);

        // Center crop, then resize to CIFAR-10 size: 32x32
        ctx.drawImage(
          img,
          sx,
          sy,
          size,
          size,
          0,
          0,
          32,
          32
        );

        const imageData = ctx.getImageData(0, 0, 32, 32).data;

        // PyTorch / ONNX expects shape: [1, 3, 32, 32]
        const input = new Float32Array(1 * 3 * 32 * 32);

        for (let i = 0; i < 32 * 32; i++) {
          let r = imageData[i * 4] / 255.0;
          let g = imageData[i * 4 + 1] / 255.0;
          let b = imageData[i * 4 + 2] / 255.0;

          // Same normalization as PyTorch transforms.Normalize(...)
          r = (r - mean[0]) / std[0];
          g = (g - mean[1]) / std[1];
          b = (b - mean[2]) / std[2];

          // Channel-first format: [R channel][G channel][B channel]
          input[i] = r;
          input[32 * 32 + i] = g;
          input[2 * 32 * 32 + i] = b;
        }

        resolve(input);
      } catch (error) {
        reject(error);
      }
    };

    img.onerror = () => {
      reject(new Error("Could not load uploaded image."));
    };

    img.src = URL.createObjectURL(file);
  });
}

function softmax(values) {
  const maxVal = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - maxVal));
  const sum = exps.reduce((a, b) => a + b, 0);

  return exps.map((v) => v / sum);
}

document.getElementById("imageInput").addEventListener("change", (event) => {
  const file = event.target.files[0];

  if (file) {
    previewImage(file);
  }
});

document.getElementById("predictBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("imageInput");
  const result = document.getElementById("result");

  try {
    if (!session) {
      result.textContent = "Model is still loading. Please try again.";
      return;
    }

    if (!fileInput.files[0]) {
      result.textContent = "Please upload an image first.";
      return;
    }

    const inputData = await preprocessImage(fileInput.files[0]);

    const inputTensor = new ort.Tensor(
      "float32",
      inputData,
      [1, 3, 32, 32]
    );

    // Use the model's actual input/output names instead of assuming "input"/"output"
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    const feeds = {};
    feeds[inputName] = inputTensor;

    const outputs = await session.run(feeds);
    const outputData = Array.from(outputs[outputName].data);

    const probabilities = softmax(outputData);

    let bestIndex = 0;

    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > probabilities[bestIndex]) {
        bestIndex = i;
      }
    }

    const confidence = (probabilities[bestIndex] * 100).toFixed(2);

    result.textContent =
      `Prediction: ${classNames[bestIndex]} (${confidence}%)`;
  } catch (error) {
    console.error("PREDICTION ERROR:", error);
    result.textContent = "Prediction failed: " + error.message;
  }
});

window.addEventListener("load", async () => {
  try {
    await loadModel();
  } catch (error) {
    console.error("FULL MODEL LOAD ERROR:", error);

    document.getElementById("modelStatus").textContent =
      "Failed to load model: " + error.message;
  }
});