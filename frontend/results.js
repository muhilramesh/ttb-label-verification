import { escapeHtml } from "./html.js";

export function createResultRenderer(resultPanel, fieldLabels) {
  function focus() {
    resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    resultPanel.focus({ preventScroll: true });
  }

  function readableError(data) {
    return data && data.error && data.error.message
      ? data.error.message : "The label could not be checked. Please try again.";
  }

  function renderError(message) {
    resultPanel.hidden = false;
    resultPanel.innerHTML = `<div class="error-panel"><h2 class="error-panel__title">Could not check label</h2><p class="error-panel__message">${escapeHtml(message)}</p></div>`;
    focus();
  }

  function renderBatchProgress(count, completed = 0) {
    resultPanel.hidden = false;
    resultPanel.innerHTML = `<div class="progress-panel"><h2>Checking labels...</h2><p>${completed} of ${count} complete</p><div class="progress-bar" role="progressbar" aria-label="Checking labels" aria-valuemin="0" aria-valuemax="${count}" aria-valuenow="${completed}"><span></span></div></div>`;
    focus();
  }

  function failureReason(result) {
    if (result.field === "government_warning") return "The warning text must match exactly, and GOVERNMENT WARNING: must be bold. Line breaks do not matter.";
    if (!result.found) return "This was not found on the label.";
    if (result.field === "abv" || result.field === "net_contents") return "The amounts do not match.";
    if (result.field === "country_of_origin") return "The countries do not match.";
    return "These do not match closely enough.";
  }

  function fieldResult(result) {
    const passed = result.status === "PASS";
    const found = result.found || "Not found on the label";
    const details = passed
      ? `<div class="comparison"><div><span class="comparison__label">Found</span><div class="comparison__value">${escapeHtml(found)}</div></div></div>`
      : `<div class="comparison"><div><span class="comparison__label">Expected</span><div class="comparison__value">${escapeHtml(result.expected || "")}</div></div><div><span class="comparison__label">Found</span><div class="comparison__value">${escapeHtml(found)}</div></div></div><p class="why-text">Why: ${escapeHtml(failureReason(result))}</p>`;
    return `<article class="field-result ${passed ? "field-result--pass" : "field-result--fail"}"><div class="field-result__header"><h3>${escapeHtml(fieldLabels[result.field] || result.field)}</h3><span class="status-badge ${passed ? "status-badge--pass" : "status-badge--fail"}">${result.status}</span></div>${details}</article>`;
  }

  function renderResults(data) {
    const approved = data.overall_verdict === "APPROVED";
    const seconds = typeof data.latency_ms === "number" ? (data.latency_ms / 1000).toFixed(1) : "0.0";
    resultPanel.hidden = false;
    resultPanel.innerHTML = `<div class="verdict ${approved ? "verdict--pass" : "verdict--review"}"><span class="verdict__label">${approved ? "APPROVED" : "NEEDS REVIEW"}</span><span class="verdict__time">Checked in ${seconds} seconds</span></div><div class="results-list">${(data.results || []).map(fieldResult).join("")}</div>`;
    focus();
  }

  function batchItem(item) {
    const passed = item.status === "APPROVED";
    const review = item.status === "NEEDS_REVIEW";
    const label = passed ? "APPROVED" : review ? "NEEDS REVIEW" : "ERROR";
    const statusClass = passed ? "status-badge--pass" : review ? "status-badge--fail" : "status-badge--error";
    const body = item.status === "ERROR"
      ? `<div class="batch-item__body"><div class="error-panel error-panel--compact"><h3>Could not check this label</h3><p>${escapeHtml(item.error?.message || "This label could not be checked.")}</p></div></div>`
      : `<div class="batch-item__body"><div class="results-list">${((item.result && item.result.results) || []).map(fieldResult).join("")}</div></div>`;
    const seconds = typeof item.latency_ms === "number" ? (item.latency_ms / 1000).toFixed(1) : "0.0";
    return `<details class="batch-item" ${passed ? "" : "open"}><summary><span><strong>${escapeHtml(item.filename || item.id)}</strong><span class="batch-item__time">${seconds} seconds</span></span><span class="status-badge ${statusClass}">${label}</span></summary>${body}</details>`;
  }

  function renderBatchResults(data) {
    const summary = data.summary || {};
    const seconds = typeof summary.latency_ms === "number" ? (summary.latency_ms / 1000).toFixed(1) : "0.0";
    resultPanel.hidden = false;
    resultPanel.innerHTML = `<div class="batch-summary"><div class="summary-tile summary-tile--pass"><span class="summary-tile__label">Approved</span><strong>${summary.passed || 0}</strong></div><div class="summary-tile summary-tile--review"><span class="summary-tile__label">Needs Review</span><strong>${summary.needs_review || 0}</strong></div><div class="summary-tile summary-tile--error"><span class="summary-tile__label">Errors</span><strong>${summary.errors || 0}</strong></div><div class="summary-tile"><span class="summary-tile__label">Total</span><strong>${summary.total || 0}</strong></div><span class="batch-summary__time">Checked in ${seconds} seconds</span></div><div class="batch-results-list">${(data.items || []).map(batchItem).join("")}</div>`;
    focus();
  }

  return { readableError, renderBatchProgress, renderBatchResults, renderError, renderResults };
}
