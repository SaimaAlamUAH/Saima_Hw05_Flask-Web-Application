document.addEventListener('DOMContentLoaded', function() {
  const imageUpload = document.getElementById('imageUpload');
  const previewImage = document.getElementById('previewImage');
  const predictBtn = document.getElementById('predictBtn');
  const predictionResult = document.getElementById('predictionResult');
  const loadingIndicator = document.getElementById('loadingIndicator');
  const errorMessage = document.getElementById('errorMessage');
  const digitElement = document.getElementById('digit');
  const confidenceElement = document.getElementById('confidence');
  const sampleDigitsContainer = document.getElementById('sampleDigits');
  let probabilityChart = null;

  // Create sample digits
  for (let i = 0; i < 10; i++) {
      const img = document.createElement('img');
      img.src = `/static/samples/${i}.png`;
      img.alt = `Sample digit ${i}`;
      img.onclick = function() {
          fetch(img.src)
              .then(response => response.blob())
              .then(blob => {
                  const file = new File([blob], `${i}.png`, { type: 'image/png' });
                  const dataTransfer = new DataTransfer();
                  dataTransfer.items.add(file);
                  imageUpload.files = dataTransfer.files;
                  handleImageSelection(file);
              });
      };
      sampleDigitsContainer.appendChild(img);
  }

  imageUpload.addEventListener('change', function() {
      if (this.files && this.files[0]) {
          handleImageSelection(this.files[0]);
      }
  });

  function handleImageSelection(file) {
      predictionResult.style.display = 'none';
      errorMessage.style.display = 'none';
      const reader = new FileReader();
      reader.onload = function(e) {
          previewImage.src = e.target.result;
          previewImage.style.display = 'block';
      };
      reader.readAsDataURL(file);
  }

  predictBtn.addEventListener('click', function() {
      if (!imageUpload.files || !imageUpload.files[0]) {
          showError('Please select an image first');
          return;
      }
      const formData = new FormData();
      formData.append('image', imageUpload.files[0]);
      loadingIndicator.style.display = 'block';
      predictionResult.style.display = 'none';
      errorMessage.style.display = 'none';
      fetch('/predict', {
          method: 'POST',
          body: formData
      })
      .then(response => response.json())
      .then(data => {
          loadingIndicator.style.display = 'none';
          if (data.error) {
              showError(data.error);
              return;
          }
          digitElement.textContent = data.digit;
          confidenceElement.textContent = data.confidence;
          drawProbabilityChart(data.probabilities);
          predictionResult.style.display = 'block';
      })
      .catch(error => {
          showError('Error: ' + error.message);
      });
  });

  function showError(message) {
      errorMessage.textContent = message;
      errorMessage.style.display = 'block';
      loadingIndicator.style.display = 'none';
  }

  function drawProbabilityChart(probabilities) {
      const ctx = document.getElementById('probabilityChart').getContext('2d');
      if (probabilityChart) probabilityChart.destroy();
      probabilityChart = new Chart(ctx, {
          type: 'bar',
          data: {
              labels: [...Array(10).keys()].map(String),
              datasets: [{
                  label: 'Probability',
                  data: probabilities,
                  backgroundColor: 'rgba(54, 162, 235, 0.5)',
                  borderColor: 'rgba(54, 162, 235, 1)',
                  borderWidth: 1
              }]
          },
          options: {
              responsive: true,
              scales: {
                  y: { beginAtZero: true, max: 1.0 }
              }
          }
      });
  }
});
