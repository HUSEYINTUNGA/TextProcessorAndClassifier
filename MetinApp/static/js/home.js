function validateAndProcess(event) {
    event.preventDefault();

    const textArea = document.querySelector('textarea[name="text"]');
    const checkboxes = document.querySelectorAll('.checkbox-group input[type="checkbox"]');
    const errorContainer = document.getElementById('errorContainer');
    errorContainer.style.display = 'none';

    if (textArea.value.trim() === '') {
        errorContainer.innerText = 'Lütfen bir metin girin.';
        errorContainer.style.display = 'block';
        return;
    }

    let isChecked = Array.from(checkboxes).some(checkbox => checkbox.checked);
    if (!isChecked) {
        errorContainer.innerText = 'Lütfen en az bir işlem seçin.';
        errorContainer.style.display = 'block';
        return;
    }


    const isUpperChecked = document.querySelector('input[name="convert_to_uppercase"]').checked;
    const isLowerChecked = document.querySelector('input[name="convert_to_lowercase"]').checked;

    if (isUpperChecked && isLowerChecked) {
        errorContainer.innerText = 'Lütfen sadece bir işlem seçin: Büyük Harfe Dönüştür veya Küçük Harfe Dönüştür.';
        errorContainer.style.display = 'block';
        return;
    }

    processAllData();
}

function processAllData() {
    const form = document.getElementById('textProcessorForm');
    const formData = new FormData(form);

    fetch(form.action, {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Bir hata oluştu.');
        }
        return response.json();
    })
    .then(data => {
        if (data.processed_text) {
            document.querySelector('.processed-text').value = data.processed_text;
            document.querySelector('.processed-text-container').style.display = 'block';
            if (data.classification_result) {
                document.getElementById('classificationText').innerText = data.classification_result;
                document.querySelector('.classification-result').style.display = 'block';
            }
        } else {
            displayError(data.error || 'Bir hata oluştu.');
        }
    })
    .catch(error => displayError(error.message));
}

function displayError(message) {
    const errorContainer = document.getElementById('errorContainer');
    errorContainer.innerText = message;
    errorContainer.style.display = 'block';
}

function copyToClipboard() {
    const processedText = document.querySelector('.processed-text');
    navigator.clipboard.writeText(processedText.value)
        .then(() => {
            const notification = document.getElementById('notification');
            notification.style.display = 'block';
            
            
            setTimeout(() => { 
                notification.style.display = 'none'; 
                const processedTextContainer = document.querySelector('.processed-text-container');
                processedTextContainer.style.display = 'none';
                const classificationResult = document.querySelector('.classification-result');
                classificationResult.style.display = 'none';
            }, 2000);
        })
        .catch(err => console.error('Panoya kopyalanamadı:', err));
}

