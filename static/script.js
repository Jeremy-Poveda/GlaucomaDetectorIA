document.getElementById('fileInput').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(event) {
            const imageElement = document.createElement('img');
            imageElement.src = event.target.result;
            document.getElementById('imageOriginal').innerHTML = '';
            document.getElementById('imageOriginal').appendChild(imageElement);
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('predictButton').addEventListener('click', () => {
    const fileInput = document.getElementById('fileInput');
    const resultDiv = document.getElementById('result');

    if (!fileInput.files.length) {
        resultDiv.textContent = 'Por favor, selecciona una imagen.';
        resultDiv.classList.add('error');
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    resultDiv.textContent = 'Cargando predicción...';
    resultDiv.classList.add('loading');

    fetch('/diagnosticate', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        resultDiv.classList.remove('loading');
        if (data.error) {
            resultDiv.textContent = 'Error: ' + data.error;
            resultDiv.classList.add('error');
            alert('La imagen no es válida como fondo de ojo.');
        } else {
            resultDiv.textContent = data.result,
            resultDiv.classList.remove('error');
            resultDiv.classList.add(data.result === 'glaucoma' ? 'success' : 'no-glaucoma');
            alert('Predicción completada');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        resultDiv.textContent = 'Hubo un error al realizar la predicción.';
        resultDiv.classList.add('error');
    });
});