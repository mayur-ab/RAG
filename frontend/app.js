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
  const profileBtn = document.getElementById("profile-btn");
  const profileModal = document.getElementById("profile-modal");
  const profileContent = document.getElementById("profile-content");
  const profileClose = document.getElementById("profile-close");
  const confirmModal = document.getElementById("confirm-modal");
  const confirmTitle = document.getElementById("confirm-title");
  const confirmMessage = document.getElementById("confirm-message");
  const confirmCancel = document.getElementById("confirm-cancel");
  const confirmOk = document.getElementById("confirm-ok");
  const reindexBtn = document.getElementById("reindex-btn");
  const docsPanel = document.getElementById("docs-panel");
  const docsList = document.getElementById("docs-list");
  const docsBrowse = document.getElementById("docs-browse");
  const docsClose = document.getElementById("docs-close");
  const docsClassifyBtn = document.getElementById("docs-classify-btn");
  const docsClusterBtn = document.getElementById("docs-cluster-btn");
  const docsClearFilterBtn = document.getElementById("docs-clear-filter-btn");
  const ingestModal = document.getElementById("ingest-modal");
  const ingestBar = document.getElementById("ingest-bar");
  const ingestStatus = document.getElementById("ingest-status");
  const ingestClose = document.getElementById("ingest-close");
  const dropZone = document.getElementById("drop-zone");
  const queryLimitHint = document.getElementById("query-limit-hint");
  const chatTitleEl = document.getElementById("chat-title");
  const sidebarRecentsEl = document.getElementById("sidebar-recents");
  const sidebarSessionsEl = document.getElementById("sidebar-sessions");
  const endSessionSidebarBtn = document.getElementById("end-session-sidebar-btn");
  const sidebar = document.getElementById("sidebar");
  const sidebarCollapseBtn = document.getElementById("sidebar-collapse-btn");
  const sidebarExpandBtn = document.getElementById("sidebar-expand-btn");
  const html = document.documentElement;

  const SIDEBAR_COLLAPSED_KEY = "rag-sidebar-collapsed";

  const CHAT_STORAGE_KEY = "rag-chat-session-v2";
  const SESSION_STORAGE_KEY = "rag-session-v2";
  const RECENTS_STORAGE_KEY = "rag-recents-v1";
  const USER_ID_STORAGE_KEY = "rag-user-id-v1";
  const MAX_STORED_RECORDS = 80;
  const MAX_RECENTS = 24;
  let maxQueryChars = 4000;
  let queryWarnChars = 3600;
  let chatPersistEnabled = true;

  const chatHistory = [];
  const messageRecords = [];
  let sessionId = null;
  let chatId = null;
  let chatCompact = "";
  let pinnedSources = [];
  let activeRecentId = null;
  let archivedSessions = [];
  let useRag = true;
  let abortController = null;
  let isGenerating = false;
  let ingestEventSource = null;
  let docsFilter = { category: null, subcategory: null, tag: null, search: "" };
  let confirmResolver = null;

  function showConfirmDialog(title, message, confirmLabel = "Confirm") {
    return new Promise((resolve) => {
      confirmResolver = resolve;
      confirmTitle.textContent = title;
      confirmMessage.textContent = message;
      confirmOk.textContent = confirmLabel;
      confirmModal.hidden = false;
      confirmModal.classList.add("modal-top");
    });
  }

  function closeConfirmDialog(result = false) {
    confirmModal.hidden = true;
    confirmModal.classList.remove("modal-top");
    if (confirmResolver) {
      confirmResolver(result);
      confirmResolver = null;
    }
  }

  function applySidebarCollapsed(collapsed) {
    sidebar?.classList.toggle("collapsed", collapsed);
    if (sidebarCollapseBtn) {
      sidebarCollapseBtn.textContent = collapsed ? "›" : "‹";
      sidebarCollapseBtn.title = collapsed ? "Expand sidebar" : "Collapse sidebar";
      sidebarCollapseBtn.setAttribute("aria-label", sidebarCollapseBtn.title);
    }
    if (sidebarExpandBtn) sidebarExpandBtn.hidden = !collapsed;
    try {
      localStorage.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? "1" : "0");
    } catch {
      // Ignore storage errors.
    }
    if (collapsed && exportMenu) exportMenu.hidden = true;
  }

  function initSidebarCollapse() {
    let collapsed = false;
    try {
      collapsed = localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1";
    } catch {
      collapsed = false;
    }
    applySidebarCollapsed(collapsed);
  }

  function closeDocsPanel() {
    docsPanel.hidden = true;
  }

  function closeProfileModal() {
    profileModal.hidden = true;
  }

  function escapeHtml(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function formatWhen(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString();
  }

  function deriveChatTitle() {
    const firstUser = messageRecords.find((m) => m.role === "user");
    if (!firstUser?.content) return "New chat";
    const text = firstUser.content.trim();
    return text.length > 52 ? `${text.slice(0, 52)}…` : text;
  }

  function getChatMessageCount() {
    return messageRecords.filter((m) => m.role === "user" || m.role === "assistant").length;
  }

  function buildCompactSnapshotForEnd() {
    const compact = (chatCompact || "").trim();
    if (compact) return compact;
    const users = messageRecords.filter((m) => m.role === "user");
    const assistants = messageRecords.filter((m) => m.role === "assistant");
    if (!users.length) return "";
    const topics = users
      .slice(0, 3)
      .map((m) => m.content.trim().slice(0, 120))
      .join("; ");
    const lastAnswer = assistants.length
      ? assistants[assistants.length - 1].content.trim().slice(0, 400)
      : "";
    if (lastAnswer) {
      return `User discussed: ${topics}. Latest assistant response covered: ${lastAnswer}`;
    }
    return `User discussed: ${topics}.`;
  }

  function formatSessionLabel(session) {
    const raw = (session.summary || "").split("\n")[0].trim();
    if (raw && !/^Session\b/i.test(raw)) {
      return raw.length > 56 ? `${raw.slice(0, 56)}…` : raw;
    }
    const count = session.chat_count || 0;
    const when = formatWhen(session.created_at || session.ended_at || "").split(",")[0];
    if (count > 0) return `${count} chat(s) - ${when || "archived"}`;
    return when ? `Session - ${when}` : "Archived session";
  }

  function dedupeRecents(recents) {
    const byId = new Map();
    for (const item of recents) {
      if (!item?.id) continue;
      const existing = byId.get(item.id);
      if (!existing || (item.updatedAt || 0) >= (existing.updatedAt || 0)) {
        byId.set(item.id, item);
      }
    }
    return [...byId.values()].sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  }

  function updateChatTitle() {
    if (chatTitleEl) chatTitleEl.textContent = deriveChatTitle();
  }

  function getRecents() {
    try {
      const raw = localStorage.getItem(RECENTS_STORAGE_KEY);
      const data = raw ? JSON.parse(raw) : [];
      return dedupeRecents(Array.isArray(data) ? data : []);
    } catch {
      return [];
    }
  }

  function saveRecents(recents) {
    try {
      localStorage.setItem(RECENTS_STORAGE_KEY, JSON.stringify(dedupeRecents(recents).slice(0, MAX_RECENTS)));
    } catch {
      // Ignore quota errors.
    }
  }

  function upsertCurrentRecent() {
    if (!messageRecords.length) return;
    const recents = getRecents();
    const id = chatId || activeRecentId || (crypto.randomUUID && crypto.randomUUID()) || `chat-${Date.now()}`;
    chatId = id;
    activeRecentId = id;
    const entry = {
      id,
      title: deriveChatTitle(),
      updatedAt: Date.now(),
      messageRecords: messageRecords.slice(-MAX_STORED_RECORDS),
      chatHistory: chatHistory.slice(-MAX_STORED_RECORDS),
      chatCompact,
      pinnedSources: pinnedSources.slice(),
      sessionId,
    };
    const idx = recents.findIndex((item) => item.id === id);
    if (idx >= 0) recents[idx] = entry;
    else recents.unshift(entry);
    saveRecents(recents);
    renderSidebarRecents();
    updateChatTitle();
  }

  function renderSidebarRecents() {
    if (!sidebarRecentsEl) return;
    const recents = getRecents();
    if (!recents.length) {
      sidebarRecentsEl.innerHTML = '<p class="profile-empty" style="padding:0.35rem 0.65rem">No chats yet</p>';
    } else {
      sidebarRecentsEl.innerHTML = recents
        .map(
          (item) =>
            `<button type="button" class="sidebar-item${item.id === activeRecentId ? " active" : ""}" data-recent-id="${escapeHtml(item.id)}" title="${escapeHtml(item.title)}">${escapeHtml(item.title)}</button>`
        )
        .join("");
      sidebarRecentsEl.querySelectorAll("[data-recent-id]").forEach((btn) => {
        btn.addEventListener("click", () => loadRecentChat(btn.dataset.recentId));
      });
    }

    if (!sidebarSessionsEl) return;
    if (!archivedSessions.length) {
      sidebarSessionsEl.innerHTML =
        '<p class="profile-empty" style="padding:0.35rem 0.65rem">End a session to archive it here</p>';
      return;
    }
    sidebarSessionsEl.innerHTML = archivedSessions
      .map((session) => {
        const short = formatSessionLabel(session);
        return `<button type="button" class="sidebar-item session-item" data-session-id="${escapeHtml(session.session_id || "")}" title="${escapeHtml(session.summary || short)}">${escapeHtml(short)}</button>`;
      })
      .join("");
    sidebarSessionsEl.querySelectorAll("[data-session-id]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const session = archivedSessions.find((s) => s.session_id === btn.dataset.sessionId);
        const detail = session?.summary || "No summary stored for this session.";
        setUploadToast(detail.length > 120 ? `${detail.slice(0, 120)}…` : detail, "");
      });
    });
  }

  async function refreshArchivedSessions() {
    try {
      const res = await fetch(`${API_BASE}/memory/profile?user_id=${encodeURIComponent(getUserId())}`);
      if (!res.ok) return;
      const profile = await res.json();
      archivedSessions = profile.recent_sessions || [];
      renderSidebarRecents();
    } catch {
      // Ignore sidebar refresh errors.
    }
  }

  function loadRecentChat(recentId) {
    const recent = getRecents().find((item) => item.id === recentId);
    if (!recent) return;
    if (isGenerating) {
      abortController?.abort();
      setGenerating(false);
      abortController = null;
    }
    hideStage();
    chatInner.innerHTML = "";
    chatHistory.length = 0;
    messageRecords.length = 0;
    recent.messageRecords?.forEach((record) => {
      if (!record?.content || !record?.role) return;
      addMessage(record.content, record.role, record.meta || {});
    });
    chatHistory.push(...(recent.chatHistory || []));
    chatCompact = recent.chatCompact || "";
    pinnedSources = Array.isArray(recent.pinnedSources) ? recent.pinnedSources.slice() : [];
    chatId = recent.id;
    activeRecentId = recent.id;
    sessionId = recent.sessionId || sessionId;
    attachEditToLastUserMessage();
    chat.scrollTop = chat.scrollHeight;
    updateChatTitle();
    renderSidebarRecents();
    persistChat();
  }

  function renderTagList(items) {
    if (!items?.length) return '<p class="profile-empty">None yet</p>';
    return items.map((item) => `<span class="profile-tag">${escapeHtml(item)}</span>`).join("");
  }

  function renderProfileModal(profile) {
    const prefs = profile.preferences || {};
    const style = profile.style || {};
    const recent = profile.recent_sessions || [];
    const active = profile.active_session || null;

    const currentHtml = active?.working_compact
      ? `<p class="profile-summary">${escapeHtml(active.working_compact)}</p>
         <div class="profile-meta">${active.chat_count || 0} chats · started ${escapeHtml(active.started_at || "")}</div>`
      : messageRecords.length
        ? `<p class="profile-empty">Active session in progress (${messageRecords.length} messages in current chat).</p>`
        : '<p class="profile-empty">No active session yet.</p>';

    const recentHtml = recent.length
      ? `<ul class="profile-list">${recent
          .map(
            (session) =>
              `<li>${escapeHtml(formatSessionLabel(session))}<div class="profile-meta">${session.chat_count || 0} chats · ${escapeHtml(
                formatWhen(session.created_at || session.ended_at || "")
              )}</div></li>`
          )
          .join("")}</ul>`
      : '<p class="profile-empty">Past sessions appear here after you click <strong>End session</strong>.</p>';

    profileContent.innerHTML = `
      <div class="profile-section">
        <div class="profile-name">${escapeHtml(profile.display_name || "Guest user")}</div>
        <div class="profile-meta">User ID: ${escapeHtml(profile.user_id || getUserId())}</div>
      </div>
      <div class="profile-section">
        <h4>Preferences</h4>
        <div class="profile-grid">
          <div class="profile-chip">Style: ${escapeHtml(prefs.response_style || "balanced")}</div>
          <div class="profile-chip">Language: ${escapeHtml(prefs.language || "english")}</div>
          <div class="profile-chip">Level: ${escapeHtml(prefs.technical_level || "intermediate")}</div>
        </div>
      </div>
      <div class="profile-section">
        <h4>Answer style</h4>
        <div class="profile-grid">
          <div class="profile-chip">${escapeHtml(style.preferred_answer_style || "balanced")}</div>
          <div class="profile-chip">${style.likes_examples ? "Likes examples" : "Examples optional"}</div>
          <div class="profile-chip">${style.likes_flowcharts ? "Likes diagrams" : "Diagrams optional"}</div>
        </div>
      </div>
      <div class="profile-section">
        <h4>Interests</h4>
        ${renderTagList(profile.interests)}
      </div>
      <div class="profile-section">
        <h4>Projects</h4>
        ${renderTagList(profile.projects)}
      </div>
      <div class="profile-section">
        <h4>Known facts</h4>
        ${renderTagList(profile.facts)}
      </div>
      <div class="profile-section">
        <h4>Frequent topics</h4>
        ${renderTagList(profile.frequent_topics)}
      </div>
      <div class="profile-section">
        <h4>Active session</h4>
        ${currentHtml}
      </div>
      <div class="profile-section">
        <h4>Recent archived sessions</h4>
        ${recentHtml}
      </div>
      <div class="profile-actions">
        <button type="button" class="toolbar-btn" id="end-session-btn">End session</button>
        <button type="button" class="toolbar-btn" id="clear-chat-btn">Clear chat history</button>
        <button type="button" class="toolbar-btn danger-btn" id="delete-memory-btn">Delete all my memory</button>
      </div>
    `;

    document.getElementById("end-session-btn")?.addEventListener("click", endSessionFromProfile);
    document.getElementById("clear-chat-btn")?.addEventListener("click", clearChatHistoryFromProfile);
    document.getElementById("delete-memory-btn")?.addEventListener("click", deleteUserMemoryFromProfile);
  }

  async function endChatAwait(compactSnapshot) {
    if (!sessionId || !chatId) return;
    const snapshot = compactSnapshot || buildCompactSnapshotForEnd();
    await fetch(`${API_BASE}/memory/session/chat-end`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: getUserId(),
        session_id: sessionId,
        chat_id: chatId,
        chat_compact: snapshot,
        message_count: getChatMessageCount(),
      }),
    }).catch(() => {});
  }

  async function endSessionFromProfile() {
    const confirmed = await showConfirmDialog(
      "End session?",
      "This merges the current chat into your session summary and archives it for long-term memory.\n\nA new session will start afterward. Indexed documents are not affected.",
      "End session"
    );
    if (!confirmed) return;

    try {
      upsertCurrentRecent();
      await ensureSession();
      const compactSnapshot = buildCompactSnapshotForEnd();
      const messageCount = getChatMessageCount();
      await endChatAwait(compactSnapshot);
      const res = await fetch(`${API_BASE}/memory/session/end`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: getUserId(),
          session_id: sessionId,
          chat_compact: compactSnapshot,
          message_count: messageCount,
          fallback_summary: compactSnapshot || deriveChatTitle(),
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `End session failed (${res.status})`);
      }
      localStorage.removeItem(SESSION_STORAGE_KEY);
      sessionId = null;
      chatCompact = "";
      newChatId();
      await ensureSession();
      await refreshArchivedSessions();
      setUploadToast("Session archived. A new session has started.", "success");
      closeProfileModal();
    } catch (err) {
      setUploadToast(err.message || "Could not end session.", "error");
    }
  }

  async function clearChatHistoryFromProfile() {
    const confirmed = await showConfirmDialog(
      "Clear chat history?",
      "This removes all messages in the current browser session and clears saved chat history from this device.\n\nYour long-term memory profile on the server is not affected.\n\nIndexed documents and RAG knowledge are not touched.",
      "Clear chat"
    );
    if (!confirmed) return;

    if (isGenerating) {
      abortController?.abort();
      setGenerating(false);
      abortController = null;
    }
    hideStage();
    endChatInBackground(chatCompact);
    chatHistory.length = 0;
    messageRecords.length = 0;
    chatCompact = "";
    pinnedSources = [];
    newChatId();
    clearPersistedChat();
    showWelcome();
    closeProfileModal();
    input.focus();
    setUploadToast("Chat history cleared on this device.", "");
  }

  async function deleteUserMemoryFromProfile() {
    const confirmed = await showConfirmDialog(
      "Delete all your memory?",
      "This permanently deletes your profile, preferences, saved name, frequent topics, session summaries, and episodic memories for this user ID.\n\nThis does NOT delete indexed documents, PDFs, chunks, or embeddings in the knowledge base.\n\nChat history in this browser is kept unless you clear it separately.",
      "Delete memory"
    );
    if (!confirmed) return;

    try {
      const res = await fetch(`${API_BASE}/memory/user?user_id=${encodeURIComponent(getUserId())}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Delete failed (${res.status})`);
      }
      setUploadToast("Your memory profile was deleted. Documents and RAG data are unchanged.", "");
      localStorage.removeItem(SESSION_STORAGE_KEY);
      sessionId = null;
      chatCompact = "";
      newChatId();
      await openProfileModal();
    } catch (err) {
      setUploadToast(err.message || "Could not delete memory.", "error");
    }
  }

  async function openProfileModal() {
    profileModal.hidden = false;
    profileContent.innerHTML = '<p class="profile-empty">Loading profile...</p>';
    try {
      const res = await fetch(`${API_BASE}/memory/profile?user_id=${encodeURIComponent(getUserId())}`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Failed to load profile (${res.status})`);
      }
      const profile = await res.json();
      archivedSessions = profile.recent_sessions || [];
      renderSidebarRecents();
      renderProfileModal(profile);
    } catch (err) {
      profileContent.innerHTML = `<p class="profile-empty">${escapeHtml(err.message || "Could not load profile.")}</p>`;
    }
  }

  function closeIngestModal() {
    ingestModal.hidden = true;
    if (ingestEventSource) {
      ingestEventSource.close();
      ingestEventSource = null;
    }
  }

  function getUserId() {
    try {
      let id = localStorage.getItem(USER_ID_STORAGE_KEY);
      if (!id) {
        id = (crypto.randomUUID && crypto.randomUUID()) || `user-${Date.now()}`;
        localStorage.setItem(USER_ID_STORAGE_KEY, id);
      }
      return id;
    } catch {
      return "anonymous-local-user";
    }
  }

  function newChatId() {
    chatId = (crypto.randomUUID && crypto.randomUUID()) || `chat-${Date.now()}`;
    chatCompact = "";
    pinnedSources = [];
    activeRecentId = chatId;
    return chatId;
  }

  function getRoutingTurns() {
    return chatHistory.slice(-4);
  }

  function persistSessionId() {
    if (!sessionId) return;
    try {
      localStorage.setItem(
        SESSION_STORAGE_KEY,
        JSON.stringify({ sessionId, userId: getUserId(), savedAt: Date.now() })
      );
    } catch {
      // Ignore storage errors.
    }
  }

  async function ensureSession() {
    if (sessionId) return sessionId;
    try {
      const raw = localStorage.getItem(SESSION_STORAGE_KEY);
      if (raw) {
        const data = JSON.parse(raw);
        if (data.sessionId && data.userId === getUserId()) {
          sessionId = data.sessionId;
          return sessionId;
        }
      }
    } catch {
      // Fall through to server start.
    }

    const res = await fetch(
      `${API_BASE}/memory/session/start?user_id=${encodeURIComponent(getUserId())}`,
      { method: "POST" }
    );
    if (!res.ok) {
      return null;
    }
    const data = await res.json();
    sessionId = data.session_id;
    persistSessionId();
    return sessionId;
  }

  function endChatInBackground(compactSnapshot) {
    if (!sessionId || !chatId) return;
    const snapshot = compactSnapshot || buildCompactSnapshotForEnd();
    fetch(`${API_BASE}/memory/session/chat-end`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: getUserId(),
        session_id: sessionId,
        chat_id: chatId,
        chat_compact: snapshot,
        message_count: getChatMessageCount(),
      }),
    }).catch(() => {
      // Best-effort merge; UI already moved on.
    });
  }

  function persistChat() {
    if (!chatPersistEnabled) return;
    upsertCurrentRecent();
    try {
      const payload = {
        chatHistory: chatHistory.slice(-MAX_STORED_RECORDS),
        messageRecords: messageRecords.slice(-MAX_STORED_RECORDS),
        chatId,
        activeRecentId,
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

      if (data.chatId) {
        chatId = data.chatId;
        activeRecentId = data.activeRecentId || data.chatId;
      } else {
        newChatId();
      }

      records.forEach((record) => {
        if (!record?.content || !record?.role) return;
        addMessage(record.content, record.role, record.meta || {});
      });

      const savedHistory = Array.isArray(data.chatHistory) ? data.chatHistory : [];
      chatHistory.push(...savedHistory.slice(-MAX_STORED_RECORDS));
      updateChatTitle();
      chatPersistEnabled = true;
      upsertCurrentRecent();
      renderSidebarRecents();
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
    if (stopBtn) {
      stopBtn.hidden = !active;
    }
    if (submitBtn) {
      submitBtn.hidden = active;
    }
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
    if (isGenerating) {
      abortController?.abort();
      setGenerating(false);
      abortController = null;
    }
    hideStage();

    upsertCurrentRecent();
    const compactSnapshot = chatCompact;
    endChatInBackground(compactSnapshot);
    chatHistory.length = 0;
    messageRecords.length = 0;
    newChatId();
    clearPersistedChat();
    showWelcome();
    updateChatTitle();
    renderSidebarRecents();
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

    if (extras.mode === "long_document" && role === "assistant") {
      const modeEl = document.createElement("div");
      modeEl.className = "meta-mode";
      const sections = extras.long_document?.sections;
      modeEl.textContent = sections
        ? `Long document · ${sections} sections (hierarchical generation)`
        : "Long document · hierarchical generation";
      div.appendChild(modeEl);
    }

    if (extras.model) {
      const meta = document.createElement("div");
      meta.className = "meta";
      const modePrefix = extras.mode === "direct" ? "direct · " : "";
      let metaText = `${modePrefix}${extras.model} · ${extras.latency?.total_seconds ?? "?"}s`;
      const tu = extras.token_usage;
      if (tu?.prompt_tokens && tu?.context_window) {
        metaText += ` · ctx ${tu.prompt_tokens.toLocaleString()}/${tu.context_window.toLocaleString()}`;
        if (tu.context_used_percent != null) {
          metaText += ` (${tu.context_used_percent}%)`;
        }
      }
      meta.textContent = metaText;
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
      await ensureSession();
      if (!chatId) newChatId();
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
      if (!sessionId) await ensureSession();
      if (!chatId) newChatId();

      const res = await fetch(`${API_BASE}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: trimmed,
          chat_compact: chatCompact || undefined,
          routing_turns: getRoutingTurns(),
          pinned_sources: pinnedSources.length ? pinnedSources : undefined,
          session_id: sessionId || undefined,
          chat_id: chatId || undefined,
          use_rag: useRag,
          use_cache: !options.skipCache,
          model: modelSelect?.value || undefined,
          user_id: getUserId(),
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
          token_usage: finalData.token_usage,
          mode: finalData.mode,
          long_document: finalData.long_document,
          cached: finalData.cached,
          grounded: finalData.grounded,
          not_in_documents: finalData.not_in_documents,
        });
        chatHistory.push({ role: "user", content: trimmed });
        chatHistory.push({ role: "assistant", content: finalData.answer });
        if (finalData.chat_compact) chatCompact = finalData.chat_compact;
        if (Array.isArray(finalData.pinned_sources) && finalData.pinned_sources.length) {
          pinnedSources = finalData.pinned_sources;
        }
        if (finalData.session_id) {
          sessionId = finalData.session_id;
          persistSessionId();
        }
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

  function renderDocsBrowse(tree) {
    if (!docsBrowse) return;
    const parts = [];
    parts.push('<input type="search" class="docs-search" id="docs-search-input" placeholder="Search documents..." />');
    parts.push(`<div class="docs-section-title">Categories (${tree.total_indexed || 0} docs)</div>`);

    const allActive = !docsFilter.category && !docsFilter.subcategory && !docsFilter.tag;
    parts.push(
      `<button type="button" class="docs-cat-btn${allActive ? " active" : ""}" data-cat="">All documents</button>`
    );
    if (tree.unclassified > 0) {
      const uncActive = docsFilter.category === "Uncategorized";
      parts.push(
        `<button type="button" class="docs-cat-btn${uncActive ? " active" : ""}" data-cat="Uncategorized">Uncategorized <span>${tree.unclassified}</span></button>`
      );
    }
    for (const cat of tree.categories || []) {
      const catActive = docsFilter.category === cat.name && !docsFilter.subcategory;
      parts.push(
        `<button type="button" class="docs-cat-btn${catActive ? " active" : ""}" data-cat="${escapeHtml(cat.name)}">📁 ${escapeHtml(cat.name)} <span>${cat.count}</span></button>`
      );
      if (docsFilter.category === cat.name) {
        for (const sub of cat.subcategories || []) {
          const subActive = docsFilter.subcategory === sub.name;
          parts.push(
            `<button type="button" class="docs-sub-btn${subActive ? " active" : ""}" data-sub="${escapeHtml(sub.name)}">${escapeHtml(sub.name)} <span>${sub.count}</span></button>`
          );
        }
      }
    }
    if (tree.tags?.length) {
      parts.push('<div class="docs-section-title">Tags</div>');
      for (const tag of tree.tags.slice(0, 20)) {
        const tagActive = docsFilter.tag === tag.name;
        parts.push(
          `<button type="button" class="docs-tag-btn${tagActive ? " active" : ""}" data-tag="${escapeHtml(tag.name)}">#${escapeHtml(tag.name)} <span>${tag.count}</span></button>`
        );
      }
    }
    docsBrowse.innerHTML = parts.join("");

    const searchInput = document.getElementById("docs-search-input");
    if (searchInput) {
      searchInput.value = docsFilter.search || "";
      searchInput.addEventListener("input", () => {
        docsFilter.search = searchInput.value.trim();
        loadDocsList();
      });
    }
    docsBrowse.querySelectorAll(".docs-cat-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        docsFilter.category = btn.dataset.cat || null;
        docsFilter.subcategory = null;
        docsFilter.tag = null;
        openDocsPanel();
      });
    });
    docsBrowse.querySelectorAll(".docs-sub-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        docsFilter.subcategory = btn.dataset.sub || null;
        docsFilter.tag = null;
        openDocsPanel();
      });
    });
    docsBrowse.querySelectorAll(".docs-tag-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        docsFilter.tag = btn.dataset.tag || null;
        docsFilter.category = null;
        docsFilter.subcategory = null;
        openDocsPanel();
      });
    });
  }

  function renderDocsList(documents) {
    if (!documents?.length) {
      docsList.innerHTML = '<div class="docs-empty">No documents match this filter.</div>';
      return;
    }
    docsList.innerHTML = documents.map((doc) => {
      const lastIngested = doc.updated_at
        ? new Date(doc.updated_at).toLocaleString()
        : "Unknown";
      const tags = (doc.tags || [])
        .map((t) => `<span class="doc-tag">#${escapeHtml(t)}</span>`)
        .join("");
      const catLabel = [doc.category, doc.subcategory].filter(Boolean).join(" › ");
      const cluster = doc.cluster_label
        ? ` · Collection: ${escapeHtml(doc.cluster_label)}`
        : "";
      return `
        <div class="doc-row">
          <div class="doc-title">${escapeHtml(doc.title || doc.source.split("/").pop())}</div>
          <div class="doc-meta">${escapeHtml(catLabel || "Uncategorized")}${cluster} · ${doc.chunk_count} chunks · ${escapeHtml(lastIngested)}</div>
          ${doc.summary ? `<div class="doc-summary">${escapeHtml(doc.summary)}</div>` : ""}
          ${tags ? `<div class="doc-tags">${tags}</div>` : ""}
          <div class="doc-path">${escapeHtml(doc.source)}</div>
        </div>
      `;
    }).join("");
  }

  async function loadDocsList() {
    const params = new URLSearchParams();
    if (docsFilter.category) params.set("category", docsFilter.category);
    if (docsFilter.subcategory) params.set("subcategory", docsFilter.subcategory);
    if (docsFilter.tag) params.set("tag", docsFilter.tag);
    if (docsFilter.search) params.set("search", docsFilter.search);
    const qs = params.toString();
    docsList.innerHTML = '<div class="docs-loading">Loading documents...</div>';
    const res = await fetch(`${API_BASE}/documents/indexed${qs ? `?${qs}` : ""}`);
    const data = await res.json();
    renderDocsList(data.documents || []);
  }

  async function openDocsPanel() {
    docsPanel.hidden = false;
    docsBrowse.innerHTML = '<div class="docs-loading">Loading categories...</div>';
    docsList.innerHTML = '<div class="docs-loading">Loading documents...</div>';
    try {
      const [treeRes] = await Promise.all([
        fetch(`${API_BASE}/documents/categories`),
        loadDocsList(),
      ]);
      const tree = await treeRes.json();
      renderDocsBrowse(tree);
    } catch (err) {
      docsBrowse.innerHTML = `<div class="docs-empty">Failed to load: ${escapeHtml(err.message)}</div>`;
    }
  }

  async function classifyAllDocuments() {
    docsClassifyBtn.disabled = true;
    docsClassifyBtn.textContent = "Classifying...";
    try {
      const res = await fetch(`${API_BASE}/documents/classify?only_missing=true`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Classification failed");
      setUploadToast(
        `Classified ${data.classified} docs (${data.skipped} skipped, ${data.failed} failed)`,
        data.failed ? "error" : "success"
      );
      await openDocsPanel();
    } catch (err) {
      setUploadToast(err.message, "error");
    } finally {
      docsClassifyBtn.disabled = false;
      docsClassifyBtn.textContent = "Classify all";
    }
  }

  async function clusterDocuments() {
    docsClusterBtn.disabled = true;
    docsClusterBtn.textContent = "Clustering...";
    try {
      const res = await fetch(`${API_BASE}/documents/cluster`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Clustering failed");
      if (!data.clustered) {
        setUploadToast(data.reason || "Not enough documents to cluster", "error");
      } else {
        setUploadToast(`Created ${data.clusters} semantic collections`, "success");
        await openDocsPanel();
      }
    } catch (err) {
      setUploadToast(err.message, "error");
    } finally {
      docsClusterBtn.disabled = false;
      docsClusterBtn.textContent = "Run clustering";
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
  initSidebarCollapse();
  setGenerating(false);

  ragToggle.addEventListener("change", () => applyRagMode(!ragToggle.checked));
  themeToggle.addEventListener("click", () => {
    applyTheme(html.getAttribute("data-theme") === "dark" ? "light" : "dark");
  });
  modelSelect?.addEventListener("change", () => selectModel(modelSelect.value));
  sidebarCollapseBtn?.addEventListener("click", () => {
    applySidebarCollapsed(!sidebar?.classList.contains("collapsed"));
  });
  sidebarExpandBtn?.addEventListener("click", () => {
    applySidebarCollapsed(false);
  });
  newChatBtn?.addEventListener("click", newChat);
  endSessionSidebarBtn?.addEventListener("click", endSessionFromProfile);
  exportBtn?.addEventListener("click", (e) => {
    e.stopPropagation();
    exportMenu.hidden = !exportMenu.hidden;
  });
  document.addEventListener("click", (e) => {
    if (!exportMenu || exportMenu.hidden) return;
    if (e.target.closest(".export-wrap")) return;
    exportMenu.hidden = true;
  });
  document.getElementById("export-md")?.addEventListener("click", exportMarkdown);
  document.getElementById("export-pdf")?.addEventListener("click", exportPdf);
  document.getElementById("export-copy")?.addEventListener("click", copyAllChat);
  docsBtn?.addEventListener("click", openDocsPanel);
  profileBtn?.addEventListener("click", openProfileModal);
  profileClose?.addEventListener("click", closeProfileModal);
  profileModal?.addEventListener("click", (e) => {
    if (e.target === profileModal) closeProfileModal();
  });
  confirmCancel?.addEventListener("click", () => closeConfirmDialog(false));
  confirmOk?.addEventListener("click", () => closeConfirmDialog(true));
  confirmModal?.addEventListener("click", (e) => {
    if (e.target === confirmModal) closeConfirmDialog(false);
  });
  docsClose?.addEventListener("click", closeDocsPanel);
  docsClassifyBtn?.addEventListener("click", classifyAllDocuments);
  docsClusterBtn?.addEventListener("click", clusterDocuments);
  docsClearFilterBtn?.addEventListener("click", () => {
    docsFilter = { category: null, subcategory: null, tag: null, search: "" };
    openDocsPanel();
  });
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
      if (!confirmModal.hidden) {
        closeConfirmDialog(false);
        return;
      }
      closeProfileModal();
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
  if (!chatId) {
    newChatId();
  }
  ensureSession().catch(() => {});
  refreshArchivedSessions().catch(() => {});
  updateChatTitle();
  renderSidebarRecents();

  checkHealth();
  loadModels();
  input.focus();
  updateQueryLimitHint();
})();
