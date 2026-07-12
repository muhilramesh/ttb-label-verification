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

const allowedTypes = new Set(["image/jpeg", "image/png", "image/webp"]);
const maxImageBytes = 8 * 1024 * 1024;
const maxBatchLabels = 10;
const requestTimeoutMs = 5000;
const batchRequestTimeoutMs = 65000;
const standardGovernmentWarning = "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic beverages during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic beverages impairs your ability to drive a car or operate machinery, and may cause health problems.";

const fieldDefinitions = [
  ["brand_name", "Brand Name", "input"],
  ["class_type", "Product Type", "input"],
  ["producer", "Producer Name", "input"],
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
  return fields.every(([, input]) => input.value.trim().length > 0);
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

  const formData = new FormData();
  const rows = batchRows.slice();
  formData.append("application_data", JSON.stringify(rows.map((row) => batchRowData(row.id))));
  formData.append("image_ids", JSON.stringify(rows.map((row) => row.id)));
  for (const row of rows) {
    formData.append("images", row.file, row.file.name);
  }

  resultPanel.hidden = true;
  resultPanel.innerHTML = "";
  setBatchLoading(true, rows.length);

  let progressTimer = setTimeout(() => renderBatchProgress(rows.length), 700);
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), batchRequestTimeoutMs);

  try {
    const response = await fetch("/verify/batch", {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    clearTimeout(progressTimer);

    const data = await response.json();
    if (!response.ok) {
      throw new Error(readableError(data));
    }

    renderBatchResults(data);
    setBatchFormMessage("");
  } catch (error) {
    clearTimeout(progressTimer);
    if (error.name === "AbortError") {
      renderError("The batch took too long to finish. Try fewer or clearer images.");
    } else {
      renderError(error.message || "The batch could not be checked. Please try again.");
    }
  } finally {
    clearTimeout(timeoutId);
    setBatchLoading(false, rows.length);
    updateBatchFormState();
  }
});

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
      ${row.previewUrl ? `<img class="batch-preview" src="${row.previewUrl}" alt="Selected label preview" />` : ""}
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
  const defaultValue = name === "government_warning" ? standardGovernmentWarning : "";
  if (type === "textarea") {
    return `
      <label class="field-label" for="${escapeHtml(inputId)}">${escapeHtml(label)}</label>
      <textarea id="${escapeHtml(inputId)}" rows="5" required>${escapeHtml(defaultValue)}</textarea>
    `;
  }
  return `
    <label class="field-label" for="${escapeHtml(inputId)}">${escapeHtml(label)}</label>
    <input id="${escapeHtml(inputId)}" type="text" autocomplete="off" required />
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
      return input && input.value.trim().length > 0;
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

function renderBatchProgress(count) {
  resultPanel.hidden = false;
  resultPanel.innerHTML = `
    <div class="progress-panel">
      <h2>Checking ${count} label${count === 1 ? "" : "s"}...</h2>
      <div class="progress-bar" role="progressbar" aria-label="Checking labels"><span></span></div>
    </div>
  `;
  focusResults();
}

function readableError(data) {
  if (data && data.error && data.error.message) {
    return data.error.message;
  }
  return "The label could not be checked. Please try again.";
}

function renderError(message) {
  resultPanel.hidden = false;
  resultPanel.innerHTML = `
    <div class="error-panel">
      <h2 class="error-panel__title">Could not check label</h2>
      <p class="error-panel__message">${escapeHtml(message)}</p>
    </div>
  `;
  focusResults();
}

function renderResults(data) {
  const approved = data.overall_verdict === "APPROVED";
  const verdictText = approved ? "APPROVED" : "NEEDS REVIEW";
  const verdictClass = approved ? "verdict--pass" : "verdict--review";
  const checkedSeconds = typeof data.latency_ms === "number" ? (data.latency_ms / 1000).toFixed(1) : "0.0";
  const fieldsHtml = (data.results || []).map(renderFieldResult).join("");

  resultPanel.hidden = false;
  resultPanel.innerHTML = `
    <div class="verdict ${verdictClass}">
      <span class="verdict__label">${verdictText}</span>
      <span class="verdict__time">Checked in ${checkedSeconds} seconds</span>
    </div>
    <div class="results-list">
      ${fieldsHtml}
    </div>
  `;
  focusResults();
}

function renderBatchResults(data) {
  const summary = data.summary || {};
  const checkedSeconds = typeof summary.latency_ms === "number" ? (summary.latency_ms / 1000).toFixed(1) : "0.0";
  const itemsHtml = (data.items || []).map(renderBatchItemResult).join("");

  resultPanel.hidden = false;
  resultPanel.innerHTML = `
    <div class="batch-summary">
      <div class="summary-tile summary-tile--pass">
        <span class="summary-tile__label">Approved</span>
        <strong>${summary.passed || 0}</strong>
      </div>
      <div class="summary-tile summary-tile--review">
        <span class="summary-tile__label">Needs Review</span>
        <strong>${summary.needs_review || 0}</strong>
      </div>
      <div class="summary-tile summary-tile--error">
        <span class="summary-tile__label">Errors</span>
        <strong>${summary.errors || 0}</strong>
      </div>
      <div class="summary-tile">
        <span class="summary-tile__label">Total</span>
        <strong>${summary.total || 0}</strong>
      </div>
      <span class="batch-summary__time">Checked in ${checkedSeconds} seconds</span>
    </div>
    <div class="batch-results-list">
      ${itemsHtml}
    </div>
  `;
  focusResults();
}

function renderBatchItemResult(item) {
  const passed = item.status === "APPROVED";
  const needsReview = item.status === "NEEDS_REVIEW";
  const statusClass = passed ? "status-badge--pass" : needsReview ? "status-badge--fail" : "status-badge--error";
  const label = passed ? "APPROVED" : needsReview ? "NEEDS REVIEW" : "ERROR";
  const open = passed ? "" : "open";
  const filename = item.filename || item.id;
  const seconds = typeof item.latency_ms === "number" ? (item.latency_ms / 1000).toFixed(1) : "0.0";
  const body = item.status === "ERROR" ? renderBatchItemError(item) : renderBatchItemDetails(item);

  return `
    <details class="batch-item" ${open}>
      <summary>
        <span>
          <strong>${escapeHtml(filename)}</strong>
          <span class="batch-item__time">${seconds} seconds</span>
        </span>
        <span class="status-badge ${statusClass}">${label}</span>
      </summary>
      ${body}
    </details>
  `;
}

function renderBatchItemError(item) {
  const message = item.error && item.error.message ? item.error.message : "This label could not be checked.";
  return `
    <div class="batch-item__body">
      <div class="error-panel error-panel--compact">
        <h3>Could not check this label</h3>
        <p>${escapeHtml(message)}</p>
      </div>
    </div>
  `;
}

function renderBatchItemDetails(item) {
  const fieldsHtml = ((item.result && item.result.results) || []).map(renderFieldResult).join("");
  return `
    <div class="batch-item__body">
      <div class="results-list">
        ${fieldsHtml}
      </div>
    </div>
  `;
}

function renderFieldResult(result) {
  const passed = result.status === "PASS";
  const statusClass = passed ? "status-badge--pass" : "status-badge--fail";
  const cardClass = passed ? "field-result--pass" : "field-result--fail";
  const fieldName = fieldLabels[result.field] || result.field;
  const found = result.found || "Not found on the label";
  const details = passed ? renderPassDetails(found) : renderFailDetails(result, found);

  return `
    <article class="field-result ${cardClass}">
      <div class="field-result__header">
        <h3>${escapeHtml(fieldName)}</h3>
        <span class="status-badge ${statusClass}">${result.status}</span>
      </div>
      ${details}
    </article>
  `;
}

function renderPassDetails(found) {
  return `
    <div class="comparison">
      <div>
        <span class="comparison__label">Found</span>
        <div class="comparison__value">${escapeHtml(found)}</div>
      </div>
    </div>
  `;
}

function renderFailDetails(result, found) {
  return `
    <div class="comparison">
      <div>
        <span class="comparison__label">Expected</span>
        <div class="comparison__value">${escapeHtml(result.expected || "")}</div>
      </div>
      <div>
        <span class="comparison__label">Found</span>
        <div class="comparison__value">${escapeHtml(found)}</div>
      </div>
    </div>
    <p class="why-text">Why: ${escapeHtml(failureReason(result))}</p>
  `;
}

function failureReason(result) {
  if (result.field === "government_warning") {
    return "The government warning must match exactly in wording, capital letters, and punctuation. Line breaks do not matter.";
  }
  if (!result.found) {
    return "This was not found on the label.";
  }
  if (result.field === "abv" || result.field === "net_contents") {
    return "The amounts do not match.";
  }
  if (result.field === "country_of_origin") {
    return "The countries do not match.";
  }
  return "These do not match closely enough.";
}

function focusResults() {
  resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  resultPanel.focus({ preventScroll: true });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function prefillStandardWarning() {
  const warningInput = document.querySelector("#government-warning");
  if (warningInput && !warningInput.value.trim()) {
    warningInput.value = standardGovernmentWarning;
  }
}

updateFormState();
updateBatchFormState();
prefillStandardWarning();
updateFormState();
