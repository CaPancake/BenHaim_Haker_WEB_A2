// Pure-JS classifier + ONNX pre-trained inference
// - Uses ONNX Runtime Web for the provided pre-trained model (animal_cnn.onnx)
// - Implements a simple fully-connected neural network trainer and predictor in plain JavaScript

const classNames = ["cat", "dog", "horse"];
const MODEL_URL = "animal_cnn.onnx";
const MODEL_KEY = "animal_cnn_onnx";

// ONNX session
let session = null;
// Pure-JS trained model
let jsModel = null;

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

async function arrayBufferToBase64(buffer) {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function base64ToUint8Array(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function loadModel() {
  const status = document.getElementById("modelStatus");
  let modelBase64 = localStorage.getItem(MODEL_KEY);

  if (!modelBase64) {
    status.textContent = "Model not in localStorage, fetching...";
    const resp = await fetch(MODEL_URL);
    if (!resp.ok) throw new Error('Failed to fetch model');
    const buffer = await resp.arrayBuffer();
    modelBase64 = await arrayBufferToBase64(buffer);
    localStorage.setItem(MODEL_KEY, modelBase64);
    status.textContent = "Model saved to localStorage.";
  } else {
    status.textContent = "Model loaded from localStorage.";
  }

  const bytes = base64ToUint8Array(modelBase64);
  session = await ort.InferenceSession.create(bytes);
  document.getElementById("modelStatus").textContent += " Ready!";
}

function previewImage(file) {
  const preview = document.getElementById("preview");
  preview.src = URL.createObjectURL(file);
  preview.style.display = 'block';
}

// Preprocess: resize to 32x32, return CHW (for ONNX) and HWC flattened for JS model
function preprocessImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    img.onload = () => {
      try {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, 32, 32);
        const data = ctx.getImageData(0, 0, 32, 32).data;

        const chw = new Float32Array(1 * 3 * 32 * 32);
        const hwc = new Float32Array(1 * 32 * 32 * 3);

        for (let i = 0; i < 32 * 32; i++) {
          const r = data[i * 4] / 255.0;
          const g = data[i * 4 + 1] / 255.0;
          const b = data[i * 4 + 2] / 255.0;
          // CHW
          chw[i] = r;
          chw[32 * 32 + i] = g;
          chw[2 * 32 * 32 + i] = b;
          // HWC
          const idx = i * 3;
          hwc[idx] = r;
          hwc[idx + 1] = g;
          hwc[idx + 2] = b;
        }

        resolve({ chw, hwc });
      } catch (e) {
        reject(e);
      }
    };

    img.onerror = () => reject(new Error('Could not load image'));
    img.src = URL.createObjectURL(file);
  });
}

function softmax(values) {
  const maxVal = Math.max(...values);
  const exps = values.map(v => Math.exp(v - maxVal));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

// ------------------ Pure-JS model (dense network) ------------------
function randArray(n, scale = 0.1) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = (Math.random() * 2 - 1) * scale;
  return a;
}

function buildJsModel({ layers, filterSize, numFilters, learningRate }) {
  const inputSize = 32 * 32 * 3;
  const hiddenUnits = Math.max(16, Math.floor(numFilters * Math.max(1, filterSize)));
  const model = { layers: [], learningRate };

  let inSize = inputSize;
  for (let i = 0; i < layers; i++) {
    const outSize = (i === layers - 1) ? Math.max(hiddenUnits, 32) : hiddenUnits;
    model.layers.push({ W: randArray(inSize * outSize, 0.05), b: new Float32Array(outSize), inSize, outSize });
    inSize = outSize;
  }

  model.output = { W: randArray(inSize * classNames.length, 0.05), b: new Float32Array(classNames.length), inSize, outSize: classNames.length };
  return model;
}

function matVecMul(mat, rows, cols, vec) {
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = 0.0;
    const base = r * cols;
    for (let c = 0; c < cols; c++) s += mat[base + c] * vec[c];
    out[r] = s;
  }
  return out;
}

function addBias(vec, bias) {
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] + bias[i];
  return out;
}

function relu(vec) {
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = Math.max(0, vec[i]);
  return out;
}

function softmaxVec(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

function predictJs(inputHwc) {
  let a = inputHwc;
  for (const layer of jsModel.layers) {
    const z = matVecMul(layer.W, layer.outSize, layer.inSize, a);
    for (let k = 0; k < z.length; k++) z[k] += layer.b[k];
    a = relu(z);
  }
  const zOut = matVecMul(jsModel.output.W, jsModel.output.outSize, jsModel.output.inSize, a);
  for (let k = 0; k < zOut.length; k++) zOut[k] += jsModel.output.b[k];
  return softmaxVec(Array.from(zOut));
}

async function trainJsModel(model, xs, ys, epochs, lr, statusEl, progressLog) {
  const n = xs.length;
  const startTime = Date.now();
  const totalIterations = epochs * n;
  let completedIterations = 0;

  const progressBar = document.getElementById('progressBar');
  const progressPercent = document.getElementById('progressPercent');
  const elapsedTimeEl = document.getElementById('elapsedTime');
  const remainingTimeEl = document.getElementById('remainingTime');

  console.log('Training started. Elements:', { progressBar, progressPercent, elapsedTimeEl, remainingTimeEl });

  function updateProgress() {
    const now = Date.now();
    const elapsed = now - startTime;
    const elapsedSec = Math.floor(elapsed / 1000);

    const percent = Math.min(100, Math.round((completedIterations / totalIterations) * 100));
    
    if (progressBar) progressBar.style.width = percent + '%';
    if (progressPercent) progressPercent.textContent = percent + '%';
    if (percent > 5 && progressBar) {
      progressBar.textContent = percent + '%';
    }

    const elapsedMinutes = Math.floor(elapsedSec / 60);
    const elapsedSeconds = elapsedSec % 60;
    if (elapsedTimeEl) {
      elapsedTimeEl.textContent = `${elapsedMinutes}:${elapsedSeconds.toString().padStart(2, '0')}`;
    }

    if (completedIterations > 0 && remainingTimeEl) {
      const avgTimePerIter = elapsed / completedIterations;
      const remainingIters = totalIterations - completedIterations;
      const remainingSec = Math.floor((avgTimePerIter * remainingIters) / 1000);
      const remainingMin = Math.floor(remainingSec / 60);
      const remainingSec2 = remainingSec % 60;
      remainingTimeEl.textContent = `${remainingMin}:${remainingSec2.toString().padStart(2, '0')}`;
    }
  }

  for (let epoch = 0; epoch < epochs; epoch++) {
    let lossSum = 0;
    let correct = 0;
    const epochStart = Date.now();

    for (let i = 0; i < n; i++) {
      const x = xs[i];
      const y = ys[i];

      const activations = [x];
      const zs = [];
      let a = x;
      for (const layer of model.layers) {
        const z = matVecMul(layer.W, layer.outSize, layer.inSize, a);
        for (let k = 0; k < z.length; k++) z[k] += layer.b[k];
        zs.push(z);
        a = relu(z);
        activations.push(a);
      }

      const zOut = matVecMul(model.output.W, model.output.outSize, model.output.inSize, a);
      for (let k = 0; k < zOut.length; k++) zOut[k] += model.output.b[k];
      const probs = softmaxVec(Array.from(zOut));

      const pred = probs.indexOf(Math.max(...probs));
      if (pred === y) correct++;

      const loss = -Math.log(Math.max(1e-7, probs[y]));
      lossSum += loss;

      const dZout = new Float32Array(probs.length);
      for (let k = 0; k < probs.length; k++) dZout[k] = probs[k] - (k === y ? 1 : 0);

      const aLast = activations[activations.length - 1];
      for (let r = 0; r < model.output.outSize; r++) {
        const db = dZout[r];
        model.output.b[r] -= lr * db;
        const base = r * model.output.inSize;
        for (let c = 0; c < model.output.inSize; c++) model.output.W[base + c] -= lr * db * aLast[c];
      }

      let dAprev = new Float32Array(model.output.inSize);
      for (let c = 0; c < model.output.inSize; c++) {
        let s = 0;
        for (let r = 0; r < model.output.outSize; r++) s += model.output.W[r * model.output.inSize + c] * dZout[r];
        dAprev[c] = s;
      }

      for (let L = model.layers.length - 1; L >= 0; L--) {
        const layer = model.layers[L];
        const z = zs[L];
        const aPrev = activations[L];

        const dZ = new Float32Array(layer.outSize);
        for (let k = 0; k < layer.outSize; k++) dZ[k] = dAprev[k] * (z[k] > 0 ? 1 : 0);

        for (let r = 0; r < layer.outSize; r++) {
          const db = dZ[r];
          layer.b[r] -= lr * db;
          const base = r * layer.inSize;
          for (let c = 0; c < layer.inSize; c++) layer.W[base + c] -= lr * db * aPrev[c];
        }

        if (L > 0) {
          const nextD = new Float32Array(layer.inSize);
          for (let c = 0; c < layer.inSize; c++) {
            let s = 0;
            for (let r = 0; r < layer.outSize; r++) s += layer.W[r * layer.inSize + c] * dZ[r];
            nextD[c] = s;
          }
          dAprev = nextD;
        }
      }

      completedIterations++;
      
      if (i % Math.max(1, Math.floor(n / 10)) === 0 || i === n - 1) {
        const avgLoss = lossSum / (i + 1);
        const acc = (correct / (i + 1) * 100).toFixed(1);
        const elapsed = ((Date.now() - epochStart) / 1000).toFixed(1);
        statusEl.textContent = `Epoch ${epoch + 1}/${epochs} — ${i + 1}/${n} samples — loss: ${avgLoss.toFixed(3)} — acc: ${acc}% — ${elapsed}s`;
      }

      if (i % Math.max(1, Math.floor(n / 20)) === 0) {
        progressLog.innerHTML += `[E${epoch + 1}:S${i}] loss=${(lossSum / (i + 1)).toFixed(3)}<br>`;
        progressLog.scrollTop = progressLog.scrollHeight;
      }
      
      updateProgress();
      
      await new Promise(res => setTimeout(res, 1));
    }

    const avgLoss = (lossSum / n).toFixed(3);
    const accuracy = (correct / n * 100).toFixed(1);
    const epochTime = ((Date.now() - epochStart) / 1000).toFixed(1);
    progressLog.innerHTML += `<strong>Epoch ${epoch + 1} done</strong> — loss: ${avgLoss} — accuracy: ${accuracy}% — time: ${epochTime}s<br>`;
    progressLog.scrollTop = progressLog.scrollHeight;
    statusEl.textContent = `Epoch ${epoch + 1}/${epochs} complete — avg loss: ${avgLoss} — accuracy: ${accuracy}%`;
    updateProgress();
  }

  const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
  progressLog.innerHTML += `Total training time: ${totalTime}s<br>`;
  updateProgress();
}

// ------------------ UI wiring ------------------

document.getElementById('imageInput').addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (f) previewImage(f);
});

document.getElementById('predictBtn').addEventListener('click', async () => {
  const fileInput = document.getElementById('imageInput');
  const result = document.getElementById('result');
  if (!fileInput.files[0]) { result.textContent = 'Please upload an image first.'; return; }

  const { chw, hwc } = await preprocessImage(fileInput.files[0]);
  const useJs = document.getElementById('useJsModel') && document.getElementById('useJsModel').checked;

  if (useJs && jsModel) {
    const probs = predictJs(hwc);
    const best = probs.indexOf(Math.max(...probs));
    result.textContent = `Prediction: ${classNames[best]} (${(probs[best]*100).toFixed(2)}%)`;
    return;
  }

  if (!session) { result.textContent = 'Model is loading...'; return; }
  const inputTensor = new ort.Tensor('float32', chw, [1, 3, 32, 32]);
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const feeds = {};
  feeds[inputName] = inputTensor;
  const outputs = await session.run(feeds);
  const out = Array.from(outputs[outputName].data);
  const probs = softmax(out);
  const best = probs.indexOf(Math.max(...probs));
  result.textContent = `Prediction: ${classNames[best]} (${(probs[best]*100).toFixed(2)}%)`;
});

async function loadTrainingDataJS(files) {
  const xs = [];
  const ys = [];

  for (const file of files) {
    const name = file.name.toLowerCase();
    const labelIndex = classNames.findIndex((c) => name.includes(c));
    if (labelIndex === -1) continue;

    const { hwc } = await preprocessImage(file);
    xs.push(hwc);
    ys.push(labelIndex);
  }

  if (xs.length === 0) throw new Error('No labeled images found. Filenames must include cat, dog, or horse.');
  return { xs, ys };
}

// Load real images from ./images folder
async function loadCifar10Sample() {
  const status = document.getElementById('trainStatus');
  const fileInput = document.getElementById('trainDataInput');
  const progressLog = document.getElementById('trainProgressLog');
  progressLog.style.display = 'block';
  progressLog.innerHTML = 'Loading real images from ./images folder...<br>';

  try {
    const dataTransfer = new DataTransfer();
    let loaded = 0;
    let failed = 0;

    // List of expected image filenames (cat, dog, horse)
    // The server will return 404 for missing files, which we skip
    const imageList = [];
    for (let i = 1; i <= 20; i++) {
      imageList.push(`cat_${i}.jpg`, `cat_${i}.png`, `dog_${i}.jpg`, `dog_${i}.png`, `horse_${i}.jpg`, `horse_${i}.png`);
    }

    for (const fileName of imageList) {
      try {
        const resp = await fetch(`./images/${fileName}`);
        if (!resp.ok) continue; // Skip if not found
        
        const blob = await resp.blob();
        const labelClass = fileName.split('_')[0]; // Extract "cat", "dog", or "horse"
        const file = new File([blob], fileName, { type: blob.type });
        dataTransfer.items.add(file);
        loaded++;
        progressLog.innerHTML += `✓ Loaded ${fileName}<br>`;
      } catch (e) {
        failed++;
      }
    }

    if (loaded === 0) {
      progressLog.innerHTML += `<strong>⚠ No images found in ./images folder</strong><br>`;
      progressLog.innerHTML += `Please create an "images" folder and add images named like: cat_1.jpg, dog_1.jpg, horse_1.jpg<br>`;
      status.textContent = 'No images found. Create ./images folder with cat_*.jpg, dog_*.jpg, horse_*.jpg files.';
      return;
    }

    fileInput.files = dataTransfer.files;
    progressLog.innerHTML += `<strong>✓ Loaded ${loaded} images. Ready to train!</strong><br>`;
    status.textContent = `Loaded ${loaded} real images. Click Train to begin.`;
  } catch (e) {
    console.error(e);
    progressLog.innerHTML += `Error: ${e.message}<br>`;
    status.textContent = 'Failed to load images';
  }
}

function createSyntheticCifarSamples() {
  // Create simple colored blocks as synthetic training images (cat=red, dog=green, horse=blue)
  const samples = [];
  const colors = [
    { r: 200, g: 50, b: 50 },   // cat - reddish
    { r: 50, g: 200, b: 50 },   // dog - greenish
    { r: 50, g: 50, b: 200 }    // horse - bluish
  ];

  for (let cls = 0; cls < 3; cls++) {
    for (let n = 0; n < 100; n++) {
      // Use regular canvas (better browser support than OffscreenCanvas)
      const canvas = document.createElement('canvas');
      canvas.width = 32;
      canvas.height = 32;
      const ctx = canvas.getContext('2d');
      
      const col = colors[cls];
      ctx.fillStyle = `rgb(${col.r}, ${col.g}, ${col.b})`;
      ctx.fillRect(0, 0, 32, 32);
      
      // Add some random noise
      const imgData = ctx.getImageData(0, 0, 32, 32);
      for (let i = 0; i < imgData.data.length; i += 4) {
        imgData.data[i] += Math.min(255, Math.max(0, imgData.data[i] + (Math.random() - 0.5) * 50));
        imgData.data[i + 1] += Math.min(255, Math.max(0, imgData.data[i + 1] + (Math.random() - 0.5) * 50));
        imgData.data[i + 2] += Math.min(255, Math.max(0, imgData.data[i + 2] + (Math.random() - 0.5) * 50));
      }
      ctx.putImageData(imgData, 0, 0);
      
      // Convert to PNG blob synchronously using canvas.toDataURL
      const dataUrl = canvas.toDataURL('image/png');
      const bstr = atob(dataUrl.split(',')[1]);
      const n_arr = new Uint8Array(bstr.length);
      for (let i = 0; i < bstr.length; i++) n_arr[i] = bstr.charCodeAt(i);
      const blob = new Blob([n_arr], { type: 'image/png' });
      
      samples.push({ blob, label: cls });
    }
  }

  return samples;
}

document.getElementById('loadCifarBtn').addEventListener('click', loadCifar10Sample);

document.getElementById('trainBtn').addEventListener('click', async () => {
  const fileInput = document.getElementById('trainDataInput');
  const status = document.getElementById('trainStatus');
  const progressLog = document.getElementById('trainProgressLog');
  const progressDiv = document.getElementById('trainProgress');
  
  if (!fileInput || !fileInput.files || fileInput.files.length === 0) { 
    status.textContent = 'Select training images (filenames must include cat/dog/horse)'; 
    return; 
  }

  progressDiv.style.display = 'block';
  progressLog.style.display = 'block';
  progressLog.innerHTML = '';

  const layers = parseInt(document.getElementById('layersInput').value, 10) || 2;
  const filterSize = parseInt(document.getElementById('filterSizeInput').value, 10) || 3;
  const numFilters = parseInt(document.getElementById('numFiltersInput').value, 10) || 16;
  const lr = parseFloat(document.getElementById('learningRateInput').value) || 0.001;
  const epochs = parseInt(document.getElementById('epochsInput').value, 10) || 5;

  status.textContent = 'Preparing data...';
  progressLog.innerHTML = 'Loading images...<br>';

  const files = Array.from(fileInput.files);
  const { xs, ys } = await loadTrainingDataJS(files);
  
  progressLog.innerHTML += `✓ Loaded ${xs.length} training samples<br>`;
  progressLog.innerHTML += `Building model (layers=${layers}, filters=${numFilters}, lr=${lr})...<br>`;
  status.textContent = `Data loaded: ${xs.length} samples. Building model...`;

  jsModel = buildJsModel({ layers, filterSize, numFilters, learningRate: lr });
  progressLog.innerHTML += `✓ Model built. Starting training...<br>`;
  
  await trainJsModel(jsModel, xs, ys, epochs, lr, status, progressLog);
  progressLog.innerHTML += `<strong>✓ Training complete!</strong><br>`;
  status.textContent = 'Training finished. Check "Use JS model" to use it for predictions.';
});

window.addEventListener('load', async () => {
  try { await loadModel(); } catch (e) { console.error(e); document.getElementById('modelStatus').textContent = 'Failed to load model.'; }
});
