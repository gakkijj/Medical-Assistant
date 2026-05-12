(function () {
  const form = document.getElementById("chatForm");
  const input = document.getElementById("messageInput");
  const button = document.getElementById("sendButton");
  const messages = document.getElementById("messages");
  const sessionBadge = document.getElementById("sessionBadge");
  const newSessionButton = document.getElementById("newSessionButton");

  const sessionKey = "medix_session_id";

  function createSessionId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return `web-${window.crypto.randomUUID()}`;
    }
    return `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function getOrCreateSessionId() {
    let current = sessionStorage.getItem(sessionKey);
    if (!current) {
      current = createSessionId();
      sessionStorage.setItem(sessionKey, current);
    }
    return current;
  }

  let sessionId = getOrCreateSessionId();

  function updateSessionBadge() {
    const shortId = sessionId.replace(/^web-/, "").slice(0, 8);
    sessionBadge.textContent = `Session: ${shortId}`;
    sessionBadge.title = sessionId;
  }

  function resetConversationView() {
    messages.innerHTML = "";
    appendMessage(
      "assistant",
      "您好，我可以帮您进行健康问题咨询、症状初步分析和生活方式建议。\n\n严重不适或紧急症状请立即就医或拨打急救电话。"
    );
  }

  function startNewSession() {
    sessionStorage.removeItem(sessionKey);
    sessionId = createSessionId();
    sessionStorage.setItem(sessionKey, sessionId);
    updateSessionBadge();
    resetConversationView();
    input.value = "";
    input.focus();
  }

  updateSessionBadge();
  resetConversationView();

  function scrollToBottom() {
    messages.scrollTop = messages.scrollHeight;
  }

  function textBlock(text) {
    const p = document.createElement("p");
    p.textContent = text || "";
    return p;
  }

  function appendMessage(role, content) {
    const article = document.createElement("article");
    article.className = `message ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    const label = document.createElement("div");
    label.className = "role-label";
    label.textContent = role === "user" ? "您" : "助手";
    bubble.appendChild(label);
    bubble.appendChild(textBlock(content));
    article.appendChild(bubble);
    messages.appendChild(article);
    scrollToBottom();
    return article;
  }

  function renderAssistantResponse(data) {
    const article = document.createElement("article");
    article.className = "message assistant";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    const label = document.createElement("div");
    label.className = "role-label";
    label.textContent = "助手";
    bubble.appendChild(label);
    bubble.appendChild(textBlock(data.answer || "抱歉，系统没有返回有效回答。"));

    if (Array.isArray(data.suggestions) && data.suggestions.length > 0) {
      const title = document.createElement("div");
      title.className = "section-title";
      title.textContent = "建议";
      bubble.appendChild(title);

      const list = document.createElement("ul");
      data.suggestions.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item;
        list.appendChild(li);
      });
      bubble.appendChild(list);
    }

    const meta = document.createElement("div");
    meta.className = "meta";
    const agents = Array.isArray(data.agents_involved) && data.agents_involved.length
      ? data.agents_involved.join(", ")
      : "未返回";
    const swarm = data.swarm_enabled ? "已启用" : "未启用";
    const coreTime = typeof data.total_time === "number" ? `${data.total_time.toFixed(2)} 秒` : "未返回";
    const totalElapsed = typeof data.client_total_time === "number"
      ? `${data.client_total_time.toFixed(2)} 秒`
      : typeof data.total_elapsed_time === "number"
        ? `${data.total_elapsed_time.toFixed(2)} 秒`
        : "未返回";
    const llmTime = typeof data.llm_total_time === "number"
      ? `${data.llm_total_time.toFixed(2)} 秒`
      : "未返回";
    const llmCount = typeof data.llm_call_count === "number" ? `${data.llm_call_count} 次` : "未返回";

    [
      `参与 Agent：${agents}`,
      `多 Agent 协作：${swarm}`,
      `总耗时：${totalElapsed}`,
      `核心处理耗时：${coreTime}`,
      `大模型调用耗时：${llmTime}`,
      `大模型调用次数：${llmCount}`,
    ].forEach((line) => {
      const row = document.createElement("div");
      row.className = "meta-row";
      row.textContent = line;
      meta.appendChild(row);
    });
    bubble.appendChild(meta);

    if (data.disclaimer) {
      const disclaimer = document.createElement("div");
      disclaimer.className = "disclaimer";
      disclaimer.textContent = data.disclaimer;
      bubble.appendChild(disclaimer);
    }

    article.appendChild(bubble);
    messages.appendChild(article);
    scrollToBottom();
  }

  async function sendMessage() {
    const message = input.value.trim();
    if (!message) {
      return;
    }

    const requestStart = performance.now();
    appendMessage("user", message);
    input.value = "";
    button.disabled = true;
    input.disabled = true;

    const loading = appendMessage("assistant", "正在分析中...");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: sessionId }),
      });

      const data = await response.json();
      data.client_total_time = (performance.now() - requestStart) / 1000;
      loading.remove();

      if (!response.ok) {
        renderAssistantResponse({
          answer: data.detail || "请求失败，请稍后重试。",
          disclaimer: "如有紧急症状，请立即就医。",
          swarm_enabled: false,
        });
        return;
      }

      if (data.session_id && data.session_id !== sessionId) {
        sessionId = data.session_id;
        sessionStorage.setItem(sessionKey, sessionId);
        updateSessionBadge();
      }

      renderAssistantResponse(data);
    } catch (error) {
      loading.remove();
      renderAssistantResponse({
        answer: "无法连接后端服务，请确认 FastAPI 服务已启动。",
        disclaimer: "本系统仅提供健康信息参考，不能替代医生诊断和治疗。",
        swarm_enabled: false,
      });
    } finally {
      button.disabled = false;
      input.disabled = false;
      input.focus();
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage();
  });

  newSessionButton.addEventListener("click", () => {
    startNewSession();
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });
})();
