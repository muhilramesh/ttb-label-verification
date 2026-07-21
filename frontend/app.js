import { createResultRenderer } from "./results.js";
import { escapeHtml } from "./html.js";

const singleModeButton = document.querySelector("#single-mode-button");
const batchModeButton = document.querySelector("#batch-mode-button");
const form = document.querySelector("#verify-form");
const batchForm = document.querySelector("#batch-form");
const imageInput = document.querySelector("#image-input");
const imagePreview = document.querySelector("#image-preview");
const fileName = document.querySelector("#file-name");
const formMessage = document.querySelector("#form-message");
const submitButton = document.querySelector("#submit-button");
const batchImageInput = document.querySelector("#batch-image-input");
const batchFileName = document.querySelector("#batch-file-name");
const batchRowsSection = document.querySelector("#batch-rows-section");
const batchRowsContainer = document.querySelector("#batch-rows");
const batchFormMessage = document.querySelector("#batch-form-message");
const batchSubmitButton = document.querySelector("#batch-submit-button");
const resultPanel = document.querySelector("#result-panel");
const batchLimit = document.querySelector("#batch-limit");

const allowedTypes = new Set(["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"]);
const maxImageBytes = 8 * 1024 * 1024;
let maxBatchLabels = 0;
let maxBatchRequestLabels = 0;
const slowMessageDelayMs = 5000;
const requestTimeoutMs = 8000;
const batchRequestTimeoutMs = 65000;
const fieldDefinitions = [
  ["brand_name", "Brand Name", "input"],
  ["class_type", "Product Type", "input"],
  ["producer", "Producer / Bottler Name and Address", "input"],
  ["country_of_origin", "Country of Origin", "input"],
  ["abv", "Alcohol Content, like 45%", "input"],
  ["net_contents", "Net Contents, like 750 mL", "input"],
  ["government_warning", "Government Warning", "textarea"],
];

const fields = [
  ["brand_name", document.querySelector("#brand-name")],
  ["class_type", document.querySelector("#product-class")],
  ["producer", document.querySelector("#producer-name")],
  ["country_of_origin", document.querySelector("#country-origin")],
  ["abv", document.querySelector("#abv")],
  ["net_contents", document.querySelector("#net-contents")],
  ["government_warning", document.querySelector("#government-warning")],
];

const fieldLabels = Object.fromEntries(
  fieldDefinitions.map(([name, label]) => [name, label.split(",")[0]]),
);
const {
  readableError,
  renderBatchProgress,
  renderBatchResults,
  renderError,
  renderResults,
} = createResultRenderer(resultPanel, fieldLabels);

let batchRows = [];

singleModeButton.addEventListener("click", () => setMode("single"));
batchModeButton.addEventListener("click", () => setMode("batch"));

function setMode(mode) {
  const batchMode = mode === "batch";
  form.hidden = batchMode;
  batchForm.hidden = !batchMode;
  singleModeButton.classList.toggle("mode-button--active", !batchMode);
  batchModeButton.classList.toggle("mode-button--active", batchMode);
  singleModeButton.setAttribute("aria-selected", String(!batchMode));
  batchModeButton.setAttribute("aria-selected", String(batchMode));
  resultPanel.hidden = true;
  resultPanel.innerHTML = "";
}

function selectedImage() {
  return imageInput.files && imageInput.files.length > 0 ? imageInput.files[0] : null;
}

function allFieldsFilled() {
  return fields.every(([, input]) => input.value.trim().length > 0 && input.checkValidity());
}

function formReady() {
  const file = selectedImage();
  return Boolean(
    file
    && allowedTypes.has(file.type)
    && file.size > 0
    && file.size <= maxImageBytes
    && allFieldsFilled(),
  );
}

function updateFormState() {
  submitButton.disabled = !formReady();
  const file = selectedImage();
  if (!file || !allFieldsFilled()) {
    setFormMessage("Choose an image and fill in all fields to check the label.");
  } else if (!allowedTypes.has(file.type)) {
    setFormMessage("Use a JPEG, PNG, or WebP image.", true);
  } else if (file.size === 0) {
    setFormMessage("That image file is empty. Choose another image.", true);
  } else if (file.size > maxImageBytes) {
    setFormMessage("Use an image smaller than 8 MB.", true);
  } else {
    setFormMessage("");
  }
}

function setFormMessage(message, isError = false, isLoading = false) {
  formMessage.textContent = message;
  formMessage.className = "form-message";
  if (isError) {
    formMessage.classList.add("form-message--error");
  }
  if (isLoading) {
    formMessage.classList.add("form-message--loading");
  }
}

function setLoading(isLoading) {
  for (const [, input] of fields) {
    input.disabled = isLoading;
  }
  form.setAttribute("aria-busy", String(isLoading));
  imageInput.disabled = isLoading;
  submitButton.disabled = isLoading || !formReady();
  submitButton.textContent = isLoading ? "Reading label..." : "Check Label";
  if (isLoading) {
    setFormMessage("Reading label...", false, true);
  }
}

function applicationData() {
  const data = {};
  for (const [name, input] of fields) {
    data[name] = name === "government_warning" ? input.value : input.value.trim();
  }
  return data;
}

function validateBeforeSubmit() {
  const file = selectedImage();
  if (!file) {
    return "Choose a label image.";
  }
  if (!allowedTypes.has(file.type)) {
    return "Use a JPEG, PNG, or WebP image.";
  }
  if (file.size === 0) {
    return "That image file is empty. Choose another image.";
  }
  if (file.size > maxImageBytes) {
    return "Use an image smaller than 8 MB.";
  }
  if (!allFieldsFilled()) {
    return "Fill in every field before checking the label.";
  }
  return "";
}

imageInput.addEventListener("change", () => {
  const file = selectedImage();
  if (!file) {
    fileName.textContent = "JPEG, PNG, or WebP";
    imagePreview.hidden = true;
    imagePreview.removeAttribute("src");
    updateFormState();
    return;
  }

  fileName.textContent = file.name;
  if (allowedTypes.has(file.type) && file.size > 0) {
    imagePreview.src = URL.createObjectURL(file);
    imagePreview.hidden = false;
  } else {
    imagePreview.hidden = true;
    imagePreview.removeAttribute("src");
  }
  updateFormState();
});

for (const [, input] of fields) {
  input.addEventListener("input", updateFormState);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const validationMessage = validateBeforeSubmit();
  if (validationMessage) {
    setFormMessage(validationMessage, true);
    return;
  }

  const formData = new FormData();
  formData.append("image", selectedImage());
  formData.append("application_data", JSON.stringify(applicationData()));

  resultPanel.hidden = true;
  resultPanel.innerHTML = "";
  setLoading(true);
  const coldStartTimer = setTimeout(() => {
    setFormMessage("Still reading. The free-tier service may be starting up...", false, true);
  }, slowMessageDelayMs);
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), requestTimeoutMs);

  try {
    const response = await fetch("/verify", {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    const data = await response.json();
    if (!response.ok) {
      throw new Error(readableError(data));
    }

    renderResults(data);
    setFormMessage("");
  } catch (error) {
    if (error.name === "AbortError") {
      renderError("The label took too long to read. Try a clearer or smaller image.");
    } else {
      renderError(error.message || "The label could not be checked. Please try again.");
    }
  } finally {
    clearTimeout(coldStartTimer);
    clearTimeout(timeoutId);
    setLoading(false);
    updateFormState();
  }
});

batchImageInput.addEventListener("change", () => {
  const files = Array.from(batchImageInput.files || []);
  const availableSlots = Math.max(0, maxBatchLabels - batchRows.length);
  const rowsToAdd = files.slice(0, availableSlots).map((file, index) => batchRowFromFile(file, index));
  batchRows = batchRows.concat(rowsToAdd);
  batchImageInput.value = "";
  renderBatchRows();
  if (files.length > availableSlots) {
    setBatchFormMessage(`Only ${maxBatchLabels} labels can be checked at a time.`, true);
    batchSubmitButton.disabled = !batchReady();
    return;
  }
  updateBatchFormState();
});

batchRowsContainer.addEventListener("input", updateBatchFormState);
batchRowsContainer.addEventListener("click", (event) => {
  const removeButton = event.target.closest("[data-remove-row]");
  if (!removeButton) {
    return;
  }
  batchRows = batchRows.filter((row) => row.id !== removeButton.dataset.removeRow);
  renderBatchRows();
  updateBatchFormState();
});

batchForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const validationMessage = validateBatchBeforeSubmit();
  if (validationMessage) {
    setBatchFormMessage(validationMessage, true);
    return;
  }

  const rows = batchRows.slice();
  resultPanel.hidden = true;
  resultPanel.innerHTML = "";
  setBatchLoading(true, rows.length);
  const started = performance.now();
  renderBatchProgress(rows.length, 0);

  try {
    const combined = {
      items: [],
      summary: { passed: 0, needs_review: 0, errors: 0, total: rows.length, latency_ms: 0 },
    };
    for (let offset = 0; offset < rows.length; offset += maxBatchRequestLabels) {
      const chunk = rows.slice(offset, offset + maxBatchRequestLabels);
      const data = await checkBatchChunk(chunk);
      combined.items.push(...(data.items || []));
      combined.summary.passed += data.summary?.passed || 0;
      combined.summary.needs_review += data.summary?.needs_review || 0;
      combined.summary.errors += data.summary?.errors || 0;
      renderBatchProgress(rows.length, combined.items.length);
    }
    combined.summary.latency_ms = Math.round(performance.now() - started);
    renderBatchResults(combined);
    setBatchFormMessage("");
  } catch (error) {
    if (error.name === "AbortError") {
      renderError("One group took too long to finish. Try again with clearer images.");
    } else {
      renderError(error.message || "The batch could not be checked. Please try again.");
    }
  } finally {
    setBatchLoading(false, rows.length);
    updateBatchFormState();
  }
});

async function checkBatchChunk(rows) {
  const formData = new FormData();
  formData.append("application_data", JSON.stringify(rows.map((row) => batchRowData(row.id))));
  formData.append("image_ids", JSON.stringify(rows.map((row) => row.id)));
  for (const row of rows) {
    formData.append("images", row.file, row.file.name);
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), batchRequestTimeoutMs);
  try {
    const response = await fetch("/verify/batch", {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(readableError(data));
    }
    return data;
  } finally {
    clearTimeout(timeoutId);
  }
}

function renderBatchRows() {
  batchRowsSection.hidden = batchRows.length === 0;
  batchFileName.textContent = batchRows.length === 0
    ? "Choose one or more JPEG, PNG, or WebP images"
    : `${batchRows.length} label image${batchRows.length === 1 ? "" : "s"} added`;

  batchRowsContainer.innerHTML = batchRows.map((row, index) => `
    <article class="batch-row" data-row-id="${escapeHtml(row.id)}">
      <div class="batch-row__header">
        <div>
          <h3>Label ${index + 1}</h3>
          <p class="batch-row__filename">${escapeHtml(row.file.name)}</p>
        </div>
        <button class="secondary-button" type="button" data-remove-row="${escapeHtml(row.id)}">Remove</button>
      </div>
      ${row.previewUrl ? `<img class="batch-preview" src="${row.previewUrl}" alt="Selected label preview" loading="lazy" decoding="async" />` : ""}
      <div class="field-grid">
        ${fieldDefinitions.map(([name, label, type]) => renderBatchInput(row.id, name, label, type)).join("")}
      </div>
    </article>
  `).join("");
}

function batchRowFromFile(file, index) {
  return {
    id: `label-${Date.now()}-${index}-${Math.random().toString(16).slice(2)}`,
    file,
    previewUrl: allowedTypes.has(file.type) && file.size > 0 ? URL.createObjectURL(file) : "",
  };
}

function renderBatchInput(rowId, name, label, type) {
  const inputId = `batch-${rowId}-${name}`;
  if (type === "textarea") {
    return `
      <label class="field-label" for="${escapeHtml(inputId)}">${escapeHtml(label)}</label>
      <textarea id="${escapeHtml(inputId)}" rows="5" required></textarea>
    `;
  }
  const numericAttrs = name === "abv"
    ? ' inputmode="decimal" pattern="[0-9]+([.][0-9]+)?([ ]?%|[ ]?(proof|Proof|PROOF))?"'
    : name === "net_contents"
      ? ' inputmode="decimal" pattern="[0-9]+([.][0-9]+)?[ ]?(mL|ml|ML|L|l|cL|cl|CL|fl oz|FL OZ|oz|OZ)"'
      : "";
  return `
    <label class="field-label" for="${escapeHtml(inputId)}">${escapeHtml(label)}</label>
    <input id="${escapeHtml(inputId)}" type="text"${numericAttrs} autocomplete="off" required />
  `;
}

function batchRowData(rowId) {
  const data = { id: rowId };
  for (const [name] of fieldDefinitions) {
    const input = document.querySelector(`#batch-${rowId}-${name}`);
    data[name] = name === "government_warning" ? input.value : input.value.trim();
  }
  return data;
}

function allBatchRowsFilled() {
  return batchRows.every((row) => {
    return fieldDefinitions.every(([name]) => {
      const input = document.querySelector(`#batch-${row.id}-${name}`);
      return input && input.value.trim().length > 0 && input.checkValidity();
    });
  });
}

function batchReady() {
  return batchRows.length > 0
    && batchRows.length <= maxBatchLabels
    && batchRows.every((row) => batchRowImagesValid(row))
    && allBatchRowsFilled();
}

function batchRowImagesValid(row) {
  return allowedTypes.has(row.file.type) && row.file.size > 0 && row.file.size <= maxImageBytes;
}

function updateBatchFormState() {
  batchSubmitButton.disabled = !batchReady();
  if (batchRows.length === 0) {
    setBatchFormMessage("Choose label images to check together.");
  } else if (batchRows.length > maxBatchLabels) {
    setBatchFormMessage(`Check at most ${maxBatchLabels} labels at a time.`, true);
  } else if (batchRows.some((row) => !allowedTypes.has(row.file.type))) {
    setBatchFormMessage("Use JPEG, PNG, or WebP images.", true);
  } else if (batchRows.some((row) => row.file.size === 0)) {
    setBatchFormMessage("One image file is empty. Remove it or choose another image.", true);
  } else if (batchRows.some((row) => row.file.size > maxImageBytes)) {
    setBatchFormMessage("Use images smaller than 8 MB.", true);
  } else if (!allBatchRowsFilled()) {
    setBatchFormMessage("Fill in every field for every label.");
  } else {
    setBatchFormMessage("");
  }
}

function validateBatchBeforeSubmit() {
  if (batchRows.length === 0) {
    return "Choose label images.";
  }
  if (batchRows.length > maxBatchLabels) {
    return `Check at most ${maxBatchLabels} labels at a time.`;
  }
  if (batchRows.some((row) => !allowedTypes.has(row.file.type))) {
    return "Use JPEG, PNG, or WebP images.";
  }
  if (batchRows.some((row) => row.file.size === 0)) {
    return "One image file is empty. Remove it or choose another image.";
  }
  if (batchRows.some((row) => row.file.size > maxImageBytes)) {
    return "Use images smaller than 8 MB.";
  }
  if (!allBatchRowsFilled()) {
    return "Fill in every field for every label.";
  }
  return "";
}

function setBatchFormMessage(message, isError = false, isLoading = false) {
  batchFormMessage.textContent = message;
  batchFormMessage.className = "form-message";
  if (isError) {
    batchFormMessage.classList.add("form-message--error");
  }
  if (isLoading) {
    batchFormMessage.classList.add("form-message--loading");
  }
}

function setBatchLoading(isLoading, count) {
  batchForm.setAttribute("aria-busy", String(isLoading));
  batchImageInput.disabled = isLoading;
  batchRowsContainer.querySelectorAll("input, textarea, button").forEach((input) => {
    input.disabled = isLoading;
  });
  batchSubmitButton.disabled = isLoading || !batchReady();
  batchSubmitButton.textContent = isLoading ? "Checking labels..." : "Check All Labels";
  if (isLoading) {
    setBatchFormMessage(`Checking ${count} label${count === 1 ? "" : "s"}...`, false, true);
  }
}


updateFormState();
updateBatchFormState();

fetch("/health")
  .then((response) => response.ok ? response.json() : null)
  .then((config) => {
    if (config && Number.isInteger(config.batch_max_labels)) {
      maxBatchRequestLabels = config.batch_max_labels;
      maxBatchLabels = Number.isInteger(config.batch_upload_max_labels)
        ? config.batch_upload_max_labels
        : config.batch_max_labels;
      batchLimit.textContent = String(maxBatchLabels);
      updateBatchFormState();
    }
  })
  .catch(() => setBatchFormMessage("Could not load the batch limit. Refresh the page.", true));

// Exercise the provider path while the user prepares the form, reducing first-submit cold starts.
fetch("/health/deep", { cache: "no-store" }).catch(() => {});
