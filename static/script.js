function generateResponse() {
    const userInput = document.getElementById("userInput").value;
    const outputBox = document.getElementById("outputBox");

    if (!userInput.trim()) {
        alert("Please enter a query.");
        return;
    }

    fetch('/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: userInput })
    })
    .then(response => response.json())
    .then(data => {
        outputBox.value = data.response;
    })
    .catch(error => {
        outputBox.value = "Error generating response.";
        console.error("Error:", error);
    });
}