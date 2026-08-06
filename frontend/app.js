(() => {
  const API_BASE = window.location.origin;

  const chat = document.getElementById("chat");
  const chatInner = document.getElementById("chat-inner");
  const form = document.getElementById("query-form");
  const input = document.getElementById("query-input");
  const submitBtn = document.getElementById("submit-btn");
  const stopBtn = document.getElementById("stop-btn");
  const attachBtn = document.getElementById("attach-btn");
  const statusEl = document.getElementById("status");
  const welcomeTpl = document.getElementById("welcome-template");
  const fileInput = document.getElementById("file-input");
  const uploadToast = document.getElementById("upload-toast");
  const themeToggle = document.getElementById("theme-toggle");
  const themeIcon = document.getElementById("theme-icon");
  const ragToggle = document.getElementById("rag-toggle");
  const ragLabel = document.getElementById("rag-label");
  const directLabel = document.getElementById("direct-label");
  const modelSelect = document.getElementById("model-select");
  const newChatBtn = document.getElementById("new-chat-btn");
  const exportBtn = document.getElementById("export-btn");
  const exportMenu = document.getElementById("export-menu");
  const docsBtn = document.getElementById("docs-btn");
  const reindexBtn = document.getElementById("reindex-btn");
  const docsPanel = document.getElementById("docs-panel");
  const docsList = document.getElementById("docs-list");
  const docsClose = document.getElementById("docs-close");
  const ingestModal = document.getElementById("ingest-modal");
  const ingestBar = document.getElementById("ingest-bar");
  const ingestStatus = document.getElementById("ingest-status");
  const ingestClose = document.getElementById("ingest-close");
  const dropZone = document.getElementById("drop-zone");
  const queryLimitHint = document.getElementById("query-limit-hint");
  const html = document.documentElement;

  const CHAT_STORAGE_KEY = "rag-chat-session-v1";
  const MAX_STORED_RECORDS = 80;
  let maxQueryChars = 4000;
  let queryWarnChars = 3600;
  let chatPersistEnabled = true;

  const chatHistory = [];
  const messageRecords = [];
  let useRag = true;
  let abortController = null;
  let isGenerating = false;
  let ingestEventSource = null;

  function closeDocsPanel() {
    docsPanel.hidden = true;
  }

  function closeIngestModal() {
    ingestModal.hidden = true;
    if (ingestEventSource) {
      ingestEventSource.close();
      ingestEventSource = null;
    }
  }

  function persistChat() {
    if (!chatPersistEnabled) return;
    try {
      const payload = {
        chatHistory: chatHistory.slice(-MAX_STORED_RECORDS),
        messageRecords: messageRecords.slice(-MAX_STORED_RECORDS),
        savedAt: Date.now(),
      };
      localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(payload));
    } catch {
      // Ignore quota errors — chat still works for the current session.
    }
  }

  function clearPersistedChat() {
    localStorage.removeItem(CHAT_STORAGE_KEY);
  }

  function restoreChat() {
    try {
      const raw = localStorage.getItem(CHAT_STORAGE_KEY);
      if (!raw) return false;

      const data = JSON.parse(raw);
      const records = Array.isArray(data.messageRecords) ? data.messageRecords : [];
      if (!records.length) return false;

      chatPersistEnabled = false;
      chatInner.innerHTML = "";
      chatHistory.length = 0;
      messageRecords.length = 0;

      records.forEach((record) => {
        if (!record?.content || !record?.role) return;
        addMessage(record.content, record.role, record.meta || {});
      });

      const savedHistory = Array.isArray(data.chatHistory) ? data.chatHistory : [];
      chatHistory.push(...savedHistory.slice(-MAX_STORED_RECORDS));
      chatPersistEnabled = true;
      attachEditToLastUserMessage();
      chat.scrollTop = chat.scrollHeight;
      return messageRecords.length > 0;
    } catch {
      chatPersistEnabled = true;
      return false;
    }
  }

  function normalizeQueryInput(raw) {
    const original = String(raw ?? "");
    let text = original.trim();
    if (!text) {
      return { text: "", wasTrimmed: false, wasOverLimit: false };
    }

    const wasOverLimit = text.length > maxQueryChars;
    if (wasOverLimit) {
      text = text.slice(0, maxQueryChars).trimEnd();
    }
    return { text, wasTrimmed: wasOverLimit, wasOverLimit };
  }

  function updateQueryLimitHint() {
    if (!queryLimitHint || !input) return;
    const length = input.value.length;

    if (length === 0) {
      queryLimitHint.textContent = "";
      queryLimitHint.className = "query-limit-hint";
      return;
    }

    queryLimitHint.textContent = `${length.toLocaleString()} / ${maxQueryChars.toLocaleString()} characters`;
    queryLimitHint.className = "query-limit-hint";
    if (length > maxQueryChars) {
      queryLimitHint.classList.add("over");
    } else if (length >= queryWarnChars) {
      queryLimitHint.classList.add("warn");
    }
  }

  function enforceInputQueryLimit(showToastOnTrim = false) {
    if (input.value.length <= maxQueryChars) {
      updateQueryLimitHint();
      return normalizeQueryInput(input.value);
    }

    input.value = input.value.slice(0, maxQueryChars);
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
    updateQueryLimitHint();
    if (showToastOnTrim) {
      setUploadToast(`Message trimmed to ${maxQueryChars.toLocaleString()} characters.`, "error");
    }
    return normalizeQueryInput(input.value);
  }

  function applyRagMode(enabled) {
    useRag = enabled;
    ragToggle.checked = !enabled;
    ragLabel.classList.toggle("active", enabled);
    directLabel.classList.toggle("active", !enabled);
    input.placeholder = enabled
      ? "Ask a question from your documents..."
      : "Chat with the model directly...";
    localStorage.setItem("rag-use-rag", enabled ? "1" : "0");
  }

  function applyTheme(theme) {
    html.setAttribute("data-theme", theme);
    themeIcon.textContent = theme === "dark" ? "☀️" : "🌙";
    localStorage.setItem("rag-theme", theme);
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderMarkdownTables(text) {
    const lines = String(text).split("\n");
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (line.includes("|") && i + 1 < lines.length && /^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$/.test(lines[i + 1])) {
        const tableLines = [line, lines[i + 1]];
        i += 2;
        while (i < lines.length && lines[i].includes("|")) {
          tableLines.push(lines[i]);
          i += 1;
        }
        const rows = tableLines.filter((_, idx) => idx !== 1).map((row) =>
          row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim())
        );
        if (rows.length) {
          const [header, ...body] = rows;
          out.push('<div class="table-wrap"><table><thead><tr>');
          header.forEach((cell) => { out.push(`<th>${escapeHtml(cell)}</th>`); });
          out.push("</tr></thead><tbody>");
          body.forEach((row) => {
            out.push("<tr>");
            row.forEach((cell) => { out.push(`<td>${escapeHtml(cell)}</td>`); });
            out.push("</tr>");
          });
          out.push("</tbody></table></div>");
          continue;
        }
      }
      out.push(escapeHtml(line));
      i += 1;
    }
    return out.join("\n").replace(/\n/g, "<br>");
  }

  function setGenerating(active) {
    isGenerating = active;
    submitBtn.disabled = active;
    attachBtn.disabled = active;
    stopBtn.hidden = !active;
    submitBtn.hidden = active;
    if (active) {
      clearUserEditActions();
    } else {
      attachEditToLastUserMessage();
    }
  }

  function clearUserEditActions() {
    chatInner.querySelectorAll(".message.user .message-actions").forEach((el) => el.remove());
  }

  function attachEditToLastUserMessage() {
    clearUserEditActions();
    if (isGenerating) return;

    const userMessages = Array.from(
      chatInner.querySelectorAll(".message.user[data-record-index]")
    );
    if (!userMessages.length) return;

    const lastUser = userMessages[userMessages.length - 1];
    if (lastUser.dataset.editing === "1") return;

    const actions = document.createElement("div");
    actions.className = "message-actions";
    const editBtn = document.createElement("button");
    editBtn.type = "button";
    editBtn.className = "msg-action-btn";
    editBtn.textContent = "Edit";
    editBtn.addEventListener("click", () => beginEditMessage(lastUser));
    actions.appendChild(editBtn);
    lastUser.appendChild(actions);
  }

  function showWelcome() {
    chatInner.innerHTML = "";
    if (welcomeTpl) {
      chatInner.appendChild(welcomeTpl.content.cloneNode(true));
      chatInner.querySelectorAll(".suggestion").forEach((btn) => {
        btn.addEventListener("click", () => askQuestion(btn.textContent));
      });
    }
  }

  function newChat() {
    if (isGenerating && abortController) abortController.abort();
    chatHistory.length = 0;
    messageRecords.length = 0;
    clearPersistedChat();
    showWelcome();
    input.focus();
  }

  function addMessage(text, role, extras = {}) {
    document.getElementById("welcome")?.remove();
    const div = document.createElement("div");
    div.className = `message ${role}`;
    div.dataset.role = role;

    const body = document.createElement("div");
    body.className = "message-body";
    if (role === "assistant" && text.includes("|")) {
      body.innerHTML = renderMarkdownTables(text);
    } else {
      body.textContent = text;
    }
    div.appendChild(body);

    if (extras.cached) {
      const cacheEl = document.createElement("div");
      cacheEl.className = "badge badge-cache";
      cacheEl.textContent = "Cached response";
      div.appendChild(cacheEl);
    }

    if (extras.not_in_documents && role === "assistant") {
      const badge = document.createElement("div");
      badge.className = "badge badge-warning";
      badge.textContent = extras.mode === "direct"
        ? "General knowledge — not from documents"
        : "Not found in documents";
      div.appendChild(badge);
    } else if (extras.grounded && role === "assistant" && extras.mode !== "direct") {
      const badge = document.createElement("div");
      badge.className = "badge badge-success";
      badge.textContent = "Grounded in documents";
      div.appendChild(badge);
    }

    if (extras.match_percent != null && role === "assistant" && extras.mode !== "direct") {
      const matchEl = document.createElement("div");
      matchEl.className = "match-label";
      if (extras.match_percent >= 75) matchEl.classList.add("high");
      else if (extras.match_percent >= 40) matchEl.classList.add("low");
      else matchEl.classList.add("weak");
      matchEl.textContent = `${extras.match_percent}% matched from documents`;
      div.appendChild(matchEl);
    }

    if (extras.mode === "direct" && role === "assistant") {
      const modeEl = document.createElement("div");
      modeEl.className = "meta-mode";
      modeEl.textContent = "Direct model — no document search";
      div.appendChild(modeEl);
    }

    if (extras.model) {
      const meta = document.createElement("div");
      meta.className = "meta";
      const modePrefix = extras.mode === "direct" ? "direct · " : "";
      meta.textContent = `${modePrefix}${extras.model} · ${extras.latency?.total_seconds ?? "?"}s`;
      div.appendChild(meta);
    }

    chatInner.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    if (role === "user" || role === "assistant") {
      const recordIndex = messageRecords.length;
      div.dataset.recordIndex = String(recordIndex);
      messageRecords.push({ role, content: text, meta: extras });
      attachEditToLastUserMessage();
      persistChat();
    }
    return div;
  }

  function createStreamingMessage() {
    document.getElementById("welcome")?.remove();
    const div = document.createElement("div");
    div.className = "message assistant";
    div.dataset.role = "assistant";
    const body = document.createElement("div");
    body.className = "message-body streaming-body";
    div.appendChild(body);
    chatInner.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return { div, body };
  }

  function showStage(message) {
    let el = document.getElementById("stage-indicator");
    if (!el) {
      el = document.createElement("div");
      el.className = "typing";
      el.id = "stage-indicator";
      chatInner.appendChild(el);
    }
    el.textContent = message;
    chat.scrollTop = chat.scrollHeight;
  }

  function hideStage() {
    document.getElementById("stage-indicator")?.remove();
  }

  function beginEditMessage(messageEl) {
    if (messageEl.dataset.editing === "1") return;

    const recordIndex = Number(messageEl.dataset.recordIndex);
    if (Number.isNaN(recordIndex) || recordIndex < 0) return;

    const body = messageEl.querySelector(".message-body");
    if (!body) return;

    clearUserEditActions();

    const originalText = body.textContent || "";
    messageEl.dataset.editing = "1";
    body.hidden = true;

    const editor = document.createElement("textarea");
    editor.className = "message-edit-input";
    editor.value = originalText;
    editor.rows = Math.min(8, Math.max(2, originalText.split("\n").length + 1));

    const editActions = document.createElement("div");
    editActions.className = "message-edit-actions";

    const cancelBtn = document.createElement("button");
    cancelBtn.type = "button";
    cancelBtn.className = "msg-action-btn edit-cancel";
    cancelBtn.textContent = "Cancel";

    const saveBtn = document.createElement("button");
    saveBtn.type = "button";
    saveBtn.className = "msg-action-btn edit-save";
    saveBtn.textContent = "Save & resend";

    const cleanup = () => {
      editor.remove();
      editActions.remove();
      body.hidden = false;
      delete messageEl.dataset.editing;
      attachEditToLastUserMessage();
    };

    cancelBtn.addEventListener("click", cleanup);

    saveBtn.addEventListener("click", async () => {
      const newText = editor.value.trim();
      if (!newText) return;

      cleanup();

      if (isGenerating && abortController) {
        abortController.abort();
        await new Promise((resolve) => {
          const waitForIdle = () => {
            if (!isGenerating) resolve();
            else setTimeout(waitForIdle, 30);
          };
          waitForIdle();
        });
      }

      Array.from(chatInner.querySelectorAll(".message[data-record-index]")).forEach((el) => {
        const idx = Number(el.dataset.recordIndex);
        if (!Number.isNaN(idx) && idx >= recordIndex) {
          el.remove();
        }
      });

      messageRecords.splice(recordIndex);
      chatHistory.length = 0;
      messageRecords.forEach((record) => {
        chatHistory.push({ role: record.role, content: record.content });
      });

      persistChat();
      await askQuestion(newText, { skipCache: true });
    });

    editActions.append(cancelBtn, saveBtn);
    messageEl.insertBefore(editor, body);
    messageEl.appendChild(editActions);
    editor.focus();
    editor.setSelectionRange(0, editor.value.length);
  }

  function setUploadToast(msg, type = "") {
    uploadToast.textContent = msg;
    uploadToast.className = `upload-toast ${type}`;
  }

  async function loadModels() {
    try {
      const res = await fetch(`${API_BASE}/models`);
      const data = await res.json();
      modelSelect.innerHTML = "";
      const current = data.current_model || "";
      (data.models || []).forEach((name) => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        if (name === current || name.startsWith(current) || current.startsWith(name)) {
          opt.selected = true;
        }
        modelSelect.appendChild(opt);
      });
      modelSelect.disabled = !data.switchable;
    } catch {
      modelSelect.innerHTML = '<option value="">Models unavailable</option>';
      modelSelect.disabled = true;
    }
  }

  async function selectModel(model) {
    if (!model) return;
    const res = await fetch(`${API_BASE}/models/select`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model }),
    });
    if (res.ok) {
      setUploadToast(`Model switched to ${model}`, "success");
    }
  }

  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      statusEl.textContent = `${data.providers.llm} · ${data.providers.vector_store}`;
      statusEl.classList.add("ok");
      if (data.limits?.max_query_chars) {
        maxQueryChars = data.limits.max_query_chars;
        queryWarnChars = data.limits.max_query_warn_chars || Math.floor(maxQueryChars * 0.9);
        updateQueryLimitHint();
      }
    } catch {
      statusEl.textContent = "Offline";
    }
  }

  async function askQuestion(query, options = {}) {
    const normalized = normalizeQueryInput(query);
    const trimmed = normalized.text;
    if (!trimmed || isGenerating) return;

    if (normalized.wasTrimmed) {
      setUploadToast(
        `Your message exceeded ${maxQueryChars.toLocaleString()} characters and was trimmed before sending.`,
        "error"
      );
    } else if (String(query).trim().length >= queryWarnChars) {
      setUploadToast(`Long message (${trimmed.length.toLocaleString()} chars) — answers may take longer.`, "");
    }

    if (!options.skipUserBubble) {
      addMessage(trimmed, "user");
    }

    input.value = "";
    input.style.height = "auto";
    setGenerating(true);
    abortController = new AbortController();

    const { div: streamDiv, body: streamBody } = createStreamingMessage();
    let streamedText = "";
    let finalData = null;

    try {
      const res = await fetch(`${API_BASE}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: trimmed,
          chat_history: chatHistory.slice(-10),
          use_rag: useRag,
          use_cache: !options.skipCache,
          model: modelSelect?.value || undefined,
        }),
        signal: abortController.signal,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        streamDiv.remove();
        addMessage(err.detail || `Request failed (${res.status})`, "error");
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data:")) continue;
          const payload = JSON.parse(line.slice(5).trim());
          if (payload.type === "stage") {
            showStage(payload.message || "Working...");
          } else if (payload.type === "token") {
            hideStage();
            streamedText += payload.content || "";
            if (streamedText.includes("|")) {
              streamBody.innerHTML = renderMarkdownTables(streamedText);
            } else {
              streamBody.textContent = streamedText;
            }
            chat.scrollTop = chat.scrollHeight;
          } else if (payload.type === "done") {
            finalData = payload.data;
          } else if (payload.type === "error") {
            throw new Error(payload.message || "Stream failed");
          }
        }
      }

      hideStage();
      streamDiv.remove();

      if (finalData) {
        addMessage(finalData.answer, "assistant", {
          match_percent: finalData.match_percent,
          model: finalData.model,
          latency: finalData.latency,
          mode: finalData.mode,
          cached: finalData.cached,
          grounded: finalData.grounded,
          not_in_documents: finalData.not_in_documents,
        });
        chatHistory.push({ role: "user", content: trimmed });
        chatHistory.push({ role: "assistant", content: finalData.answer });
        persistChat();
      } else if (streamedText) {
        addMessage(streamedText, "assistant", { mode: useRag ? "rag" : "direct" });
        chatHistory.push({ role: "user", content: trimmed });
        chatHistory.push({ role: "assistant", content: streamedText });
        persistChat();
      }
    } catch (err) {
      hideStage();
      streamDiv?.remove();
      if (err.name === "AbortError") {
        if (streamedText) {
          addMessage(streamedText + "\n\n[Stopped]", "assistant", { mode: useRag ? "rag" : "direct" });
          chatHistory.push({ role: "user", content: trimmed });
          chatHistory.push({ role: "assistant", content: streamedText });
          persistChat();
        } else {
          addMessage("Generation stopped.", "system");
        }
      } else {
        addMessage(`Could not reach the API: ${err.message}`, "error");
      }
    } finally {
      setGenerating(false);
      abortController = null;
      input.focus();
    }
  }

  function buildMarkdownExport() {
    const lines = ["# RAG Chat Export", "", `Exported: ${new Date().toLocaleString()}`, ""];
    messageRecords.forEach((msg) => {
      const label = msg.role === "user" ? "You" : "Assistant";
      lines.push(`## ${label}`, "", msg.content, "");
    });
    return lines.join("\n");
  }

  function exportMarkdown() {
    const blob = new Blob([buildMarkdownExport()], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `rag-chat-${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
    exportMenu.hidden = true;
  }

  function exportPdf() {
    const printWindow = window.open("", "_blank");
    const body = messageRecords.map((msg) => {
      const label = msg.role === "user" ? "You" : "Assistant";
      return `<section><h3>${escapeHtml(label)}</h3><pre>${escapeHtml(msg.content)}</pre></section>`;
    }).join("");
    printWindow.document.write(`<!DOCTYPE html><html><head><title>RAG Chat Export</title>
      <style>body{font-family:Segoe UI,sans-serif;padding:24px;max-width:800px;margin:0 auto}
      h3{margin:1rem 0 0.25rem}pre{white-space:pre-wrap;background:#f5f5f5;padding:12px;border-radius:8px}</style>
      </head><body><h1>RAG Chat Export</h1>${body}</body></html>`);
    printWindow.document.close();
    printWindow.focus();
    printWindow.print();
    exportMenu.hidden = true;
  }

  async function copyAllChat() {
    const text = buildMarkdownExport();
    await navigator.clipboard.writeText(text);
    setUploadToast("Chat copied to clipboard", "success");
    exportMenu.hidden = true;
  }

  async function uploadFiles(files) {
    if (!files?.length) return;
    attachBtn.disabled = true;
    submitBtn.disabled = true;
    const list = Array.from(files);
    setUploadToast(`Uploading ${list.length} file(s)...`);
    const formData = new FormData();
    list.forEach((file) => formData.append("files", file));

    try {
      const res = await fetch(`${API_BASE}/upload/bulk?folder=Uploaded-Docs&auto_ingest=true`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Upload failed (${res.status})`);
      }
      const data = await res.json();
      setUploadToast(`Uploaded ${data.success}/${data.total} file(s)`, "success");
      (data.results || []).forEach((item) => {
        if (item.status === "ok") {
          const chunks = item.ingestion?.total_chunks ?? 0;
          addMessage(`"${item.filename}" added to knowledge base (${chunks} chunks).`, "system");
        } else {
          addMessage(`Upload failed for ${item.filename}: ${item.error}`, "error");
        }
      });
    } catch (err) {
      setUploadToast(err.message, "error");
      addMessage(`Upload failed: ${err.message}`, "error");
    } finally {
      attachBtn.disabled = false;
      submitBtn.disabled = false;
      fileInput.value = "";
    }
  }

  async function openDocsPanel() {
    docsPanel.hidden = false;
    docsList.innerHTML = '<div class="docs-loading">Loading indexed documents...</div>';
    try {
      const res = await fetch(`${API_BASE}/documents/indexed`);
      const data = await res.json();
      if (!data.documents?.length) {
        docsList.innerHTML = '<div class="docs-empty">No indexed documents yet.</div>';
        return;
      }
      docsList.innerHTML = data.documents.map((doc) => {
        const lastIngested = doc.updated_at
          ? new Date(doc.updated_at).toLocaleString()
          : "Unknown";
        return `
        <div class="doc-row">
          <div class="doc-title">${escapeHtml(doc.title || doc.source.split("/").pop())}</div>
          <div class="doc-meta">${doc.chunk_count} chunks · ${escapeHtml(doc.status)} · Last indexed: ${escapeHtml(lastIngested)}</div>
          <div class="doc-path">${escapeHtml(doc.source)}</div>
        </div>
      `;
      }).join("");
    } catch (err) {
      docsList.innerHTML = `<div class="docs-empty">Failed to load: ${escapeHtml(err.message)}</div>`;
    }
  }

  function startFolderIngest() {
    if (ingestEventSource) {
      ingestEventSource.close();
    }
    ingestModal.hidden = false;
    ingestBar.style.width = "0%";
    ingestStatus.textContent = "Starting bulk ingest...";
    ingestEventSource = new EventSource(`${API_BASE}/ingest/folder/stream`);

    ingestEventSource.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "start") {
        ingestStatus.textContent = `Ingesting 0 / ${payload.total} files...`;
      } else if (payload.type === "progress") {
        const pct = Math.round((payload.current / payload.total) * 100);
        ingestBar.style.width = `${pct}%`;
        ingestStatus.textContent = `[${payload.current}/${payload.total}] ${payload.filename} — ${payload.status}`;
      } else if (payload.type === "done") {
        ingestBar.style.width = "100%";
        ingestStatus.textContent = `Done — ingested: ${payload.success}, skipped: ${payload.skipped}, failed: ${payload.failed}, vectors: ${payload.total_vectors}`;
        ingestEventSource?.close();
        ingestEventSource = null;
      }
    };

    ingestEventSource.onerror = () => {
      ingestStatus.textContent = "Ingest stream disconnected. You can close this dialog.";
      ingestEventSource?.close();
      ingestEventSource = null;
    };
  }

  // Events
  const savedRagMode = localStorage.getItem("rag-use-rag");
  applyRagMode(savedRagMode !== "0");
  applyTheme(localStorage.getItem("rag-theme") ||
    (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"));

  ragToggle.addEventListener("change", () => applyRagMode(!ragToggle.checked));
  themeToggle.addEventListener("click", () => {
    applyTheme(html.getAttribute("data-theme") === "dark" ? "light" : "dark");
  });
  modelSelect?.addEventListener("change", () => selectModel(modelSelect.value));
  newChatBtn?.addEventListener("click", newChat);
  exportBtn?.addEventListener("click", () => { exportMenu.hidden = !exportMenu.hidden; });
  document.getElementById("export-md")?.addEventListener("click", exportMarkdown);
  document.getElementById("export-pdf")?.addEventListener("click", exportPdf);
  document.getElementById("export-copy")?.addEventListener("click", copyAllChat);
  docsBtn?.addEventListener("click", openDocsPanel);
  docsClose?.addEventListener("click", closeDocsPanel);
  docsPanel?.addEventListener("click", (e) => {
    if (e.target === docsPanel) closeDocsPanel();
  });
  reindexBtn?.addEventListener("click", startFolderIngest);
  ingestClose?.addEventListener("click", closeIngestModal);
  ingestModal?.addEventListener("click", (e) => {
    if (e.target === ingestModal) closeIngestModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeIngestModal();
      closeDocsPanel();
      if (exportMenu) exportMenu.hidden = true;
    }
  });
  stopBtn?.addEventListener("click", () => abortController?.abort());

  attachBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => uploadFiles(fileInput.files));
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const normalized = enforceInputQueryLimit(true);
    askQuestion(normalized.text);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
  });
  input.addEventListener("input", () => {
    if (input.value.length > maxQueryChars) {
      input.value = input.value.slice(0, maxQueryChars);
      setUploadToast(`Maximum ${maxQueryChars.toLocaleString()} characters — extra text removed.`, "error");
    }
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 140) + "px";
    updateQueryLimitHint();
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropZone?.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
  });
  ["dragleave", "drop"].forEach((evt) => {
    dropZone?.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.remove("drag-over"); });
  });
  dropZone?.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));

  document.querySelectorAll(".suggestion").forEach((btn) => {
    btn.addEventListener("click", () => askQuestion(btn.textContent));
  });

  if (!restoreChat()) {
    showWelcome();
  }

  checkHealth();
  loadModels();
  input.focus();
  updateQueryLimitHint();
})();
