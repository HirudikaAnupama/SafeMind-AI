document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("user-input").addEventListener("keypress", function (event) {
        if (event.key === "Enter") {
            sendMessage();
        }
    });
});

function sendMessage() {
    let userInput = document.getElementById("user-input").value.trim();
    if (userInput === "") return;

    appendMessage(userInput, "user-message");
    document.getElementById("user-input").value = "";

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: userInput })
    })
    .then(response => response.json())
    .then(data => {
        if (data.response) {  // ✅ Fix here
            appendMessage(data.response, "bot-message");
        } else {
            appendMessage("Sorry, I couldn't understand.", "bot-message");
        }
    })
    .catch(error => {
        console.error("Error:", error);
        appendMessage("Error connecting to the server.", "bot-message");
    });
}

function appendMessage(text, className) {
    let chatBox = document.getElementById("chat-box");
    let messageDiv = document.createElement("div");
    messageDiv.classList.add("message", className);
    messageDiv.textContent = text;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}



