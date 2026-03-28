/* custom.js — global utilities only. Page-specific logic lives in each template. */

document.addEventListener('DOMContentLoaded', function () {

    // ── Upload page (legacy analyzeBtn flow) ─────────────────────────────────
    // Only runs when the old analyzeBtn element is present (non-upload pages).
    const analyzeBtn   = document.getElementById('analyzeBtn');
    const form         = document.getElementById('uploadForm');
    const resultSection = document.getElementById('resultSection');

    if (analyzeBtn && form) {
        const dropZones    = document.querySelectorAll('.drop-zone');
        let uploadedFiles  = new Map();

        function updateSubmitButton() {
            const requiredInputs = document.querySelectorAll('input[required]');
            const allFilled = Array.from(requiredInputs).every(input => {
                if (input.type === 'file') return uploadedFiles.has(input.dataset.view);
                return input.value.trim() !== '';
            });
            analyzeBtn.disabled = !allFilled;
        }

        dropZones.forEach(zone => {
            const fileInput        = zone.querySelector('input[type="file"]');
            const previewContainer = zone.querySelector('.preview-container');
            const previewImage     = previewContainer?.querySelector('.preview-image');
            const uploadContent    = zone.querySelector('.upload-content');
            const removeButton     = zone.querySelector('.remove-image');
            const view             = fileInput?.dataset.view;

            if (!fileInput || !view) return;

            zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('dragover'); });
            zone.addEventListener('dragleave', ()  => zone.classList.remove('dragover'));

            zone.addEventListener('drop', e => {
                e.preventDefault();
                zone.classList.remove('dragover');
                const file = e.dataTransfer.files[0];
                if (file && file.type.startsWith('image/')) handleFile(file, view);
            });

            zone.addEventListener('click', () => {
                if (!uploadedFiles.has(view)) fileInput.click();
            });

            fileInput.addEventListener('change', () => {
                if (fileInput.files[0]) handleFile(fileInput.files[0], view);
            });

            if (removeButton) {
                removeButton.addEventListener('click', e => {
                    e.stopPropagation();
                    uploadedFiles.delete(view);
                    fileInput.value = '';
                    previewContainer?.classList.add('hidden');
                    uploadContent?.classList.remove('hidden');
                    updateSubmitButton();
                });
            }

            function handleFile(file, view) {
                if (!file || !file.type.startsWith('image/')) return;
                uploadedFiles.set(view, file);
                const reader = new FileReader();
                reader.onload = e => {
                    if (previewImage)  previewImage.src = e.target.result;
                    previewContainer?.classList.remove('hidden');
                    uploadContent?.classList.add('hidden');
                };
                reader.readAsDataURL(file);
                updateSubmitButton();
            }
        });

        form.addEventListener('submit', async e => {
            e.preventDefault();

            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Analyzing...';

            const formData = new FormData();
            formData.append('brand_name', document.getElementById('brandName')?.value || '');

            uploadedFiles.forEach((file, view) => {
                formData.append('images[]', file);
                formData.append('view_types[]', view);
            });

            try {
                const response = await fetch('/api/detect/', {
                    method: 'POST',
                    body: formData,
                    headers: {
                        'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]')?.value || ''
                    }
                });

                const result = await response.json();

                if (resultSection) {
                    resultSection.classList.remove('hidden');
                    resultSection.innerHTML = `
                        <div class="p-6 rounded-lg shadow-lg" style="background-color:var(--card-color);color:var(--text-color)">
                            <h4 class="text-lg font-semibold mb-4">Analysis Complete</h4>
                            <div class="space-y-4">
                                <span class="font-bold text-xl ${result.is_fake ? 'text-red-500' : 'text-green-500'}">
                                    ${result.is_fake ? 'FAKE' : 'AUTHENTIC'}
                                </span>
                                <p>${result.message || ''}</p>
                                <p class="text-sm">Confidence: ${((result.confidence || 0) * 100).toFixed(1)}%</p>
                            </div>
                        </div>`;
                    resultSection.scrollIntoView({ behavior: 'smooth' });
                }

                if (result.id) {
                    setTimeout(() => {
                        window.location.href = result.redirect === 'admin'
                            ? result.admin_url
                            : `/result/${result.id}/`;
                    }, 1500);
                }

            } catch (err) {
                console.error('Upload error:', err);
                const errDiv = document.createElement('div');
                errDiv.className = 'text-red-500 mt-4';
                errDiv.textContent = 'An error occurred. Please try again.';
                form.appendChild(errDiv);
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = 'Analyze All Images';
            }
        });

        // Set initial button state
        updateSubmitButton();
    }

    // ── Mobile menu toggle (present on every page via base.html) ─────────────
    const mobileMenuBtn = document.getElementById('mobile-menu-button');
    const mobileMenu    = document.getElementById('mobile-menu');
    if (mobileMenuBtn && mobileMenu) {
        mobileMenuBtn.addEventListener('click', () => mobileMenu.classList.toggle('hidden'));
    }

});
