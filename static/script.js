document.getElementById('fileInput').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (event) {
            const imageElement = document.createElement('img');
            imageElement.src = event.target.result;

            imageElement.style.maxWidth = '100%';
            imageElement.style.maxHeight = '100%';
            imageElement.style.objectFit = 'contain';

            const imageContainer = document.getElementById('imageOriginal');
            imageContainer.innerHTML = ''; 
            imageContainer.appendChild(imageElement); 
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('predictButton').addEventListener('click', () => {
    const fileInput = document.getElementById('fileInput');
    const resultDiv = document.getElementById('result');

    if (!fileInput.files.length) {
        showModal('Por favor, selecciona una imagen.');
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    resultDiv.textContent = 'Cargando predicción...';

    fetch('/diagnosticate', {
        method: 'POST',
        body: formData,
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                resultDiv.textContent = '';
                showModal('Error: ' + data.error);
            } else {
                resultDiv.textContent = data.result;
                showModal('Resultado: ' + data.result);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showModal('Hubo un error al realizar la predicción.');
        });
});

function showModal(message) {
    const modal = document.getElementById('customModal');
    const modalMessage = document.getElementById('modalMessage');
    const closeBtn = document.querySelector('.close-btn');

    modalMessage.textContent = message;
    modal.classList.add('show'); 

    closeBtn.addEventListener('click', () => {
        modal.classList.remove('show'); 
    });

    window.onclick = function (event) {
        if (event.target === modal) {
            modal.classList.remove('show');
        }
    };
}

