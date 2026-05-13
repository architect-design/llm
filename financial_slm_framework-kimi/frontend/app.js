/**
 * Financial SLM Framework - Frontend Application
 * Vanilla JavaScript SPA for file validation and generation
 */

const API_BASE = window.location.origin.includes('localhost') 
    ? 'http://localhost:8000/api' 
    : '/api';

// State
const state = {
    currentView: 'validate',
    specs: [],
    validationResult: null,
    generatedContent: null,
    isLoading: false
};

// DOM Elements
const elements = {};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    cacheElements();
    bindEvents();
    loadSpecs();
    loadModelStatus();
});

function cacheElements() {
    // Navigation
    elements.navItems = document.querySelectorAll('.nav-item');
    elements.views = document.querySelectorAll('.view');

    // Validation
    elements.validateSpecSelect = document.getElementById('validate-spec-select');
    elements.dropZone = document.getElementById('drop-zone');
    elements.fileInput = document.getElementById('file-input');
    elements.pasteContent = document.getElementById('paste-content');
    elements.validatePasteBtn = document.getElementById('validate-paste-btn');
    elements.resultsContent = document.getElementById('results-content');
    elements.resultsSummary = document.getElementById('results-summary');
    elements.resultBadge = document.getElementById('result-badge');
    elements.summaryRecords = document.getElementById('summary-records');
    elements.summaryErrors = document.getElementById('summary-errors');
    elements.summaryWarnings = document.getElementById('summary-warnings');
    elements.summaryChecksum = document.getElementById('summary-checksum');

    // Generation
    elements.generateSpecSelect = document.getElementById('generate-spec-select');
    elements.numRecords = document.getElementById('num-records');
    elements.genSeed = document.getElementById('gen-seed');
    elements.generateBtn = document.getElementById('generate-btn');
    elements.generatedContent = document.getElementById('generated-content');
    elements.generationMeta = document.getElementById('generation-meta');
    elements.genStatus = document.getElementById('gen-status');
    elements.genChecksum = document.getElementById('gen-checksum');
    elements.copyBtn = document.getElementById('copy-btn');
    elements.downloadBtn = document.getElementById('download-btn');

    // Specs
    elements.specsGrid = document.getElementById('specs-grid');

    // Model
    elements.statParams = document.getElementById('stat-params');
    elements.statVocab = document.getElementById('stat-vocab');
    elements.statDevice = document.getElementById('stat-device');
    elements.trainSpecSelect = document.getElementById('train-spec-select');
    elements.trainSamples = document.getElementById('train-samples');
    elements.trainEpochs = document.getElementById('train-epochs');
    elements.trainBtn = document.getElementById('train-btn');
    elements.trainingLog = document.getElementById('training-log');
}

function bindEvents() {
    // Navigation
    elements.navItems.forEach(item => {
        item.addEventListener('click', () => switchView(item.dataset.view));
    });

    // File upload
    elements.dropZone.addEventListener('click', () => elements.fileInput.click());
    elements.dropZone.addEventListener('dragover', handleDragOver);
    elements.dropZone.addEventListener('dragleave', handleDragLeave);
    elements.dropZone.addEventListener('drop', handleDrop);
    elements.fileInput.addEventListener('change', handleFileSelect);

    // Validation
    elements.validatePasteBtn.addEventListener('click', validatePastedContent);

    // Generation
    elements.generateBtn.addEventListener('click', generateFile);
    elements.copyBtn.addEventListener('click', copyToClipboard);
    elements.downloadBtn.addEventListener('click', downloadGenerated);

    // Training
    elements.trainBtn.addEventListener('click', startTraining);
}

// Navigation
function switchView(viewName) {
    state.currentView = viewName;

    elements.navItems.forEach(item => {
        item.classList.toggle('active', item.dataset.view === viewName);
    });

    elements.views.forEach(view => {
        view.classList.toggle('active', view.id === `view-${viewName}`);
    });
}

// API Helpers
async function apiGet(endpoint) {
    const response = await fetch(`${API_BASE}${endpoint}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
}

async function apiPost(endpoint, data) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || `HTTP ${response.status}`);
    }
    return response.json();
}

async function apiPostForm(endpoint, formData) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        body: formData
    });
    if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || `HTTP ${response.status}`);
    }
    return response.json();
}

// Load Specifications
async function loadSpecs() {
    try {
        state.specs = await apiGet('/specs');
        populateSpecSelects();
        renderSpecsGrid();
    } catch (error) {
        showToast('Failed to load specifications', 'error');
        console.error(error);
    }
}

function populateSpecSelects() {
    const options = state.specs.map(s => 
        `<option value="${s.spec_id}">${s.name}</option>`
    ).join('');

    const defaultOption = '<option value="">Select format...</option>';

    elements.validateSpecSelect.innerHTML = defaultOption + options;
    elements.generateSpecSelect.innerHTML = defaultOption + options;
    elements.trainSpecSelect.innerHTML = defaultOption + options;
}

function renderSpecsGrid() {
    elements.specsGrid.innerHTML = state.specs.map(spec => `
        <div class="spec-card" onclick="showSpecDetail('${spec.spec_id}')">
            <div class="spec-card-header">
                <h3>${spec.name}</h3>
                <span class="spec-version">${spec.version}</span>
            </div>
            <p class="spec-description">${spec.description}</p>
            <div class="spec-records">
                ${spec.record_types.map(rt => 
                    `<span class="spec-record-tag" title="${rt.name}">${rt.code} (${rt.length})</span>`
                ).join('')}
            </div>
        </div>
    `).join('');
}

// File Upload Handlers
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    elements.dropZone.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    elements.dropZone.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    elements.dropZone.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

async function processFile(file) {
    const specId = elements.validateSpecSelect.value;
    if (!specId) {
        showToast('Please select a specification first', 'error');
        return;
    }

    setLoading(true);

    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('spec_id', specId);

        const result = await apiPostForm('/validate/upload', formData);
        renderValidationResult(result);
        showToast('Validation complete', 'success');
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        setLoading(false);
    }
}

async function validatePastedContent() {
    const specId = elements.validateSpecSelect.value;
    const content = elements.pasteContent.value;

    if (!specId) {
        showToast('Please select a specification first', 'error');
        return;
    }
    if (!content.trim()) {
        showToast('Please paste file content', 'error');
        return;
    }

    setLoading(true);

    try {
        const result = await apiPost('/validate', {
            content: content,
            spec_id: specId,
            filename: 'pasted_content'
        });
        renderValidationResult(result);
        showToast('Validation complete', 'success');
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        setLoading(false);
    }
}

// Render Validation Results
function renderValidationResult(result) {
    state.validationResult = result;

    // Update badge
    elements.resultBadge.className = `result-badge ${result.overall_status}`;
    elements.resultBadge.textContent = result.overall_status.toUpperCase();

    // Update summary
    elements.resultsSummary.style.display = 'block';
    elements.summaryRecords.textContent = result.summary.total_records;
    elements.summaryErrors.textContent = result.summary.total_errors;
    elements.summaryWarnings.textContent = result.summary.total_warnings;
    elements.summaryChecksum.textContent = result.checksum_valid ? 'Valid' : 'Invalid';
    elements.summaryChecksum.className = `summary-value ${result.checksum_valid ? '' : 'error'}`;

    // Render record cards
    const recordsHtml = result.records.map((rec, idx) => {
        const hasIssues = rec.results.length > 0;
        const isExpanded = hasIssues;

        return `
            <div class="record-card">
                <div class="record-card-header" onclick="toggleRecordDetails(${idx})">
                    <div class="record-info">
                        <span class="record-type-badge">${rec.record_type}</span>
                        <span class="record-number">#${rec.record_number}</span>
                    </div>
                    <span class="record-status ${rec.status}">${rec.status}</span>
                </div>
                <div class="record-details ${isExpanded ? 'expanded' : ''}" id="record-details-${idx}">
                    ${rec.results.length > 0 ? rec.results.map(r => `
                        <div class="result-item">
                            <div class="result-icon ${r.severity}">${r.severity === 'error' ? '!' : '?'}</div>
                            <div class="result-content">
                                <div class="result-field">${r.field_name}</div>
                                <div class="result-message">${r.message}</div>
                                <div class="result-meta">
                                    Expected: "${r.expected}" | Actual: "${r.actual}" | Pos: ${r.position[0]}-${r.position[1]}
                                </div>
                            </div>
                        </div>
                    `).join('') : '<div class="result-item"><div class="result-content"><div class="result-message" style="color: var(--accent-green)">All fields valid</div></div></div>'}
                </div>
            </div>
        `;
    }).join('');

    elements.resultsContent.innerHTML = recordsHtml;
}

function toggleRecordDetails(idx) {
    const details = document.getElementById(`record-details-${idx}`);
    if (details) {
        details.classList.toggle('expanded');
    }
}

// Generation
async function generateFile() {
    const specId = elements.generateSpecSelect.value;
    const numRecords = parseInt(elements.numRecords.value) || 10;
    const useSlm = document.querySelector('input[name="gen-mode"]:checked').value === 'slm';
    const seed = elements.genSeed.value ? parseInt(elements.genSeed.value) : null;

    if (!specId) {
        showToast('Please select a specification', 'error');
        return;
    }

    setLoading(true);
    elements.generateBtn.disabled = true;
    elements.generateBtn.innerHTML = '<div class="spinner"></div> Generating...';

    try {
        const result = await apiPost('/generate', {
            spec_id: specId,
            num_records: numRecords,
            use_slm: useSlm,
            seed: seed
        });

        state.generatedContent = result.content;
        renderGeneratedContent(result.content);

        elements.generationMeta.style.display = 'flex';
        elements.genStatus.textContent = `Status: ${result.validation_status}`;
        elements.genStatus.style.color = result.validation_status === 'valid' ? 'var(--accent-green)' : 'var(--accent-amber)';
        elements.genChecksum.textContent = `Checksum: ${result.checksum_valid ? 'Valid' : 'Invalid'}`;

        showToast('File generated successfully', 'success');
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        setLoading(false);
        elements.generateBtn.disabled = false;
        elements.generateBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M13 10V3L4 14h7v7l9-11h-7z"/>
            </svg>
            Generate File
        `;
    }
}

function renderGeneratedContent(content) {
    const lines = content.split('\n');
    const html = lines.map((line, idx) => `
        <div class="line">
            <span class="line-number">${idx + 1}</span>
            <span class="line-content">${escapeHtml(line)}</span>
        </div>
    `).join('');

    elements.generatedContent.innerHTML = html;
}

function copyToClipboard() {
    if (!state.generatedContent) return;
    navigator.clipboard.writeText(state.generatedContent).then(() => {
        showToast('Copied to clipboard', 'success');
    });
}

function downloadGenerated() {
    if (!state.generatedContent) return;

    const specId = elements.generateSpecSelect.value;
    const blob = new Blob([state.generatedContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `generated_${specId}_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);

    showToast('File downloaded', 'success');
}

// Model Status
async function loadModelStatus() {
    try {
        const status = await apiGet('/model/status');
        elements.statParams.textContent = status.total_parameters.toLocaleString();
        elements.statVocab.textContent = status.vocab_size.toLocaleString();
        elements.statDevice.textContent = status.device;
    } catch (error) {
        elements.statParams.textContent = 'Not loaded';
        elements.statVocab.textContent = 'Not loaded';
        elements.statDevice.textContent = 'Not loaded';
    }
}

// Training
async function startTraining() {
    const specId = elements.trainSpecSelect.value;
    const samples = parseInt(elements.trainSamples.value) || 1000;
    const epochs = parseInt(elements.trainEpochs.value) || 5;

    if (!specId) {
        showToast('Please select a specification', 'error');
        return;
    }

    elements.trainBtn.disabled = true;
    elements.trainBtn.textContent = 'Training...';
    elements.trainingLog.innerHTML = '<div class="log-entry">Starting training...</div>';

    try {
        const result = await apiPost('/train', {
            spec_id: specId,
            num_samples: samples,
            epochs: epochs,
            batch_size: 16
        });

        elements.trainingLog.innerHTML += `
            <div class="log-entry" style="color: var(--accent-green)">
                Training complete! Final loss: ${result.final_train_loss?.toFixed(4) || 'N/A'}
            </div>
            <div class="log-entry">Checkpoint saved to: ${result.checkpoint_dir}</div>
        `;

        showToast('Training completed successfully', 'success');
        loadModelStatus();
    } catch (error) {
        elements.trainingLog.innerHTML += `
            <div class="log-entry" style="color: var(--accent-red)">Error: ${error.message}</div>
        `;
        showToast(error.message, 'error');
    } finally {
        elements.trainBtn.disabled = false;
        elements.trainBtn.textContent = 'Start Training';
    }
}

// Utility Functions
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span class="toast-message">${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(20px)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function setLoading(loading) {
    state.isLoading = loading;
    document.body.style.cursor = loading ? 'wait' : 'default';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showSpecDetail(specId) {
    // Could open a modal with detailed spec view
    showToast(`Specification: ${specId}`, 'info');
}
