const imageInput = document.getElementById("imageInput");
const predictButton = document.getElementById("predictButton");

const previewContainer =
    document.getElementById("previewContainer");

const loading =
    document.getElementById("loading");

const result =
    document.getElementById("result");


imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (!file) {
        return;
    }

    const imageURL = URL.createObjectURL(file);

    previewContainer.innerHTML = `
        <img src="${imageURL}" alt="Vehicle">
    `;

});


predictButton.addEventListener("click", async function () {

    const file = imageInput.files[0];

    if (!file) {

        alert("Please select an image.");

        return;
    }

    const formData = new FormData();

    formData.append("file", file);

    loading.style.display = "block";

    result.innerHTML = "";

    try {

        const response = await fetch(
            "/predict",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        displayResults(data);

    } catch (error) {

        result.innerHTML = `
            <p>
                Error connecting to the server.
            </p>
        `;

    } finally {

        loading.style.display = "none";

    }

});


function displayResults(data) {

    const predictions = data.predictions;

    let html = `
        <div class="result-card">

            <h2>
                Prediction: ${predictions[0].class}
            </h2>

            <h3>
                Confidence:
                ${(predictions[0].confidence * 100).toFixed(2)}%
            </h3>

            <hr>

            <h3>Top 3 Predictions</h3>
    `;

    predictions.forEach((prediction, index) => {

        html += `
            <div class="prediction">

                <span>
                    ${index + 1}.
                    ${prediction.class}
                </span>

                <strong>
                    ${(prediction.confidence * 100).toFixed(2)}%
                </strong>

            </div>
        `;

    });

    html += `
        </div>
    `;

    result.innerHTML = html;
}