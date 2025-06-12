// detector/static/detector/js/script.js
document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
    const button = form.querySelector('button');
    const fileInput = form.querySelector('input[type=\"file\"]');

    form.addEventListener('submit', function () {
        button.disabled = true;
        button.textContent = 'Detecting...';
    });

    fileInput.addEventListener('change', function () {
        if (fileInput.files.length > 0) {
            button.disabled = false;
            button.textContent = 'Detect';
        }
    });
});
