document.getElementById("chatForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  
  const queryInput = document.getElementById("query");
  const query = queryInput.value.trim();
  if (!query) return;

  const chatBox = document.getElementById("chatBox");

  const userMsg = document.createElement("div");
  userMsg.className = "message user";
  userMsg.textContent = query;
  chatBox.appendChild(userMsg);
  
  queryInput.value = "";
  chatBox.scrollTop = chatBox.scrollHeight;

  const loadingMsg = document.createElement("div");
  loadingMsg.className = "message ai";
  loadingMsg.innerHTML = "<strong>SupportIQ:</strong> <em>Typing...</em>";
  chatBox.appendChild(loadingMsg);

  try {
    const res = await fetch("/ask", { 
        method: "POST", 
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query }) 
    });

    const data = await res.json();
    chatBox.removeChild(loadingMsg);

    const botMsg = document.createElement("div");
    botMsg.className = "message ai";
    botMsg.innerHTML = `
        <strong>SupportIQ:</strong><br>${data.answer}<br>
        <span class="level-tag level-${data.level}">Source: ${data.level} Support</span>
    `;
    
    chatBox.appendChild(botMsg);
    chatBox.scrollTop = chatBox.scrollHeight;

  } catch (err) {
    chatBox.removeChild(loadingMsg);
    console.error(err);
  }
});