const classNames = ["cat", "dog", "horse"];

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

async function loadModel() {
  const status = document.getElementById("modelStatus");

  let modelBase64 = localStorage.getItem("animal_cnn_onnx");

  if (!modelBase64) {
    status.textContent = "Model not found in localStorage.";

    const response = await fetch("animal_cnn.onnx");

    if (!response.ok) {
      throw new Error("Could not fetch model file.");
    }

    const buffer = await response.arrayBuffer();

    modelBase64 = await arrayBufferToBase64(buffer);
    localStorage.setItem("animal_cnn_onnx", modelBase64);

    status.textContent = "Model saved to localStorage.";
  } else {
    status.textContent = "Model loaded from localStorage.";
  }

  const modelBytes = base64ToUint8Array(modelBase64);

  session = await ort.InferenceSession.create(modelBytes);

  status.textContent += " Ready!";
}

function previewImage(file) {
  const preview = document.getElementById("preview");

  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
}

function preprocessImage(file) {
  return new Promise((resolve) => {
    const img = new Image();
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    const mean = [0.4914, 0.4822, 0.4465];
    const std = [0.2470, 0.2435, 0.2616];

    img.onload = () => {
      // Center crop before resizing to 32x32
      // since the model was built on CIFAR10 with 32x32 images piped
      const size = Math.min(img.width, img.height);
      const sx = (img.width - size) / 2;
      const sy = (img.height - size) / 2;

      ctx.drawImage(
        img,
        sx, sy, size, size,  // crop area from original image
        0, 0, 32, 32         // draw into 32x32 canvas
      );

      const imageData = ctx.getImageData(0, 0, 32, 32).data;

      // PyTorch expects shape: [1, 3, 32, 32]
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
    };

    img.src = URL.createObjectURL(file);
  });
}

function softmax(values) {
  const maxVal = Math.max(...values);
  const exps = values.map(v => Math.exp(v - maxVal));
  const sum = exps.reduce((a, b) => a + b, 0);

  return exps.map(v => v / sum);
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

  const feeds = {
    input: inputTensor
  };

  const outputs = await session.run(feeds);
  const outputData = Array.from(outputs.output.data);

  const probabilities = softmax(outputData);

  let bestIndex = 0;

  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > probabilities[bestIndex]) {
      bestIndex = i;
    }
  }

  const confidence = (probabilities[bestIndex] * 100).toFixed(2);

  result.textContent = `Prediction: ${classNames[bestIndex]} (${confidence}%)`;
});

window.addEventListener("load", async () => {
  try {
    await loadModel();
  } catch (error) {
    console.error(error);
    document.getElementById("modelStatus").textContent =
      "Failed to load model.";
  }
});