document.addEventListener("DOMContentLoaded", () => {
    const csvForm = document.getElementById("csv-form");
    const columnsForm = document.getElementById("columns-form");
    const textProcessorForm = document.getElementById("textProcessorForm");

    const showError = (element, message) => {
        const errorElement = element.querySelector(".error-message");
        if (errorElement) {
            errorElement.textContent = message;
            errorElement.style.display = "block";
        }
    };

    const clearError = (element) => {
        const errorElement = element.querySelector(".error-message");
        if (errorElement) {
            errorElement.textContent = "";
            errorElement.style.display = "none";
        }
    };

    if (csvForm) {
        csvForm.addEventListener("submit", (event) => {
            const fileInput = document.getElementById("csvFile");

            if (!fileInput.value) {
                showError(csvForm, "Lütfen bir dosya seçin.");
                event.preventDefault();
            } else {
                const file = fileInput.files[0];
                const fileName = file.name;
                const fileSize = file.size;

                if (!fileName.endsWith(".csv")) {
                    showError(csvForm, "Lütfen yalnızca .csv uzantılı dosyalar yükleyin.");
                    event.preventDefault();
                } else if (fileSize > 10 * 1024 * 1024) {
                    showError(csvForm, "Dosya boyutu çok büyük. Lütfen 10MB'den küçük bir dosya yükleyin.");
                    event.preventDefault();
                } else {
                    clearError(csvForm);
                }
            }
        });
    }

    if (columnsForm) {
        columnsForm.addEventListener("submit", (event) => {
            const selectedColumns = Array.from(
                columnsForm.querySelectorAll('input[name="selected_columns"]:checked')
            );

            if (selectedColumns.length === 0) {
                showError(columnsForm, "Lütfen en az bir sütun seçin.");
                event.preventDefault();
            } else {
                clearError(columnsForm);
                columnsForm.style.display = "none";
                console.log("Seçilen Sütunlar:", selectedColumns.map(column => column.value));
            }
        });
    }


    if (textProcessorForm) {
        textProcessorForm.addEventListener("submit", (event) => {
            const selectedOptions = Array.from(
                textProcessorForm.querySelectorAll("input[type='checkbox']:checked")
            );

            if (selectedOptions.length === 0) {
                showError(textProcessorForm, "Lütfen en az bir işlem seçin.");
                event.preventDefault();
            } else {
                clearError(textProcessorForm);

                
                console.log("Seçilen İşlemler:", selectedOptions.map(option => option.name));
            }
        });
    }
});
