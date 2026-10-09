const sessionsView = document.querySelector("#sessions");
const messagesView = document.querySelector("#messages");
const chatTitle = document.querySelector("#chat-title");
const statusView = document.querySelector("#status");
const composer = document.querySelector("#composer");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const newButton = document.querySelector("#new-chat");
const compareButton = document.querySelector("#compare-button");
const configurationSelect = document.querySelector("#configuration-select");
const configurationStatus = document.querySelector("#configuration-status");
const comparisonSelect = document.querySelector("#comparison-select");
let referenceConfiguration = "baseline";
const memoryToggle = document.querySelector("#memory-toggle");
let configurations = [];
let defaultConfiguration = "optimized";
let polling = false;
let currentSession = null;
let busy = false;
let models = [];
let timer = null;
const modelSelect = document.querySelector("#model-select");
const modelStatus = document.querySelector("#model-status");
function modelLabel(provider, model) {
  return `${provider === "ollama" ? "Локальная" : "DeepSeek"}${model ? ` · ${model}` : ""}`;
}
function renderModel() {
  const selected = currentSession?.provider || "ollama";
  const config = configurations.find((c) => c.id === selectedConfiguration());
  modelSelect.replaceChildren(...models.map((model) => {
    const option = element("option", "", modelLabel(model.id, model.id === "ollama" ? config?.model || model.model : model.model));
    option.value = model.id;
    option.disabled = model.id === "ollama" ? !configurations.some((c) => c.available) : !model.available;
    return option;
  }));
  modelSelect.value = selected;
  const model = models.find((item) => item.id === selected);
  modelStatus.textContent = selected === "ollama" && config
    ? config.available ? "Локальная модель готова" : "Выбранная модель не установлена"
    : model?.detail || "Проверяю модели…";
  renderConfiguration();
}

function selectedConfiguration() {
  return currentSession?.configuration || defaultConfiguration;
}

function renderConfiguration() {
  const local = currentSession?.provider === "ollama";
  configurationSelect.replaceChildren(...configurations.map((config) => {
    const option = element("option", "", config.label + (config.available ? "" : " · модель не установлена"));
    option.value = config.id;
    option.disabled = !config.available;
    return option;
  }));
  configurationSelect.value = selectedConfiguration();
  configurationSelect.disabled = !local;
  const config = configurations.find((c) => c.id === selectedConfiguration());
  configurationStatus.textContent = local && config
    ? `${config.model} · temperature ${config.profile.temperature} · контекст ${config.profile.num_ctx} · ответ до ${config.profile.max_tokens} токенов`
    : "Настройки оптимизации доступны для локальной модели";
  if (referenceConfiguration === selectedConfiguration()) {
    referenceConfiguration = selectedConfiguration() === "baseline" ? "optimized" : "baseline";
  }
  comparisonSelect.replaceChildren(...configurations.filter((c) => c.id !== selectedConfiguration()).map((config) => {
    const option = element("option", "", `Сравнить с: ${config.label}`);
    option.value = config.id;
    option.disabled = !config.available;
    return option;
  }));
  comparisonSelect.value = referenceConfiguration;
  comparisonSelect.disabled = !local || busy;
  compareButton.textContent = referenceConfiguration === "baseline" ? "Сравнить с исходным"
    : referenceConfiguration === "optimized" ? "Сравнить с оптимизированным" : "Сравнить варианты";
  updateComposer();
}
comparisonSelect.addEventListener("change", () => {
  referenceConfiguration = comparisonSelect.value;
  renderConfiguration();
});

configurationSelect.addEventListener("change", async () => {
  if (!currentSession) return;
  try {
    currentSession = await api(`/api/rag-chat/sessions/${currentSession.id}/configuration`, {
      method: "PUT", body: JSON.stringify({ configuration: configurationSelect.value }),
    });
    renderModel();
  } catch (error) {
    statusView.textContent = error.message;
    renderModel();
  }
});

function showMemory(visible) {
  document.querySelector(".shell").classList.toggle("memory-hidden", !visible);
  memoryToggle.setAttribute("aria-expanded", String(visible));
  memoryToggle.textContent = visible ? "Скрыть память задачи" : "Показать память задачи";
}
memoryToggle.addEventListener("click", () => showMemory(memoryToggle.getAttribute("aria-expanded") !== "true"));
modelSelect.addEventListener("change", async () => {
  if (!currentSession) return;
  const id = currentSession.id;
  try {
    currentSession = await api(`/api/rag-chat/sessions/${id}/model`, {
      method: "PUT", body: JSON.stringify({ provider: modelSelect.value }),
    });
    renderModel();
    renderMessages();
    updateComposer();
  } catch (error) {
    statusView.textContent = error.message;
    renderModel();
  }
});

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : "Не удалось выполнить запрос");
  }
  return response.status === 204 ? null : response.json();
}

function element(tag, className = "", content = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = content;
  return node;
}

function sourceLink(source) {
  const link = element("a");
  if (source.kind === "message") {
    link.href = `#turn-${source.message_id}`;
    link.textContent = source.title || "Сообщение пользователя";
    link.addEventListener("click", () => {
      const target = document.getElementById(`turn-${source.message_id}`);
      target?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  } else {
    link.href = `/api/document-index/source#page=${source.page_start}`;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = `${source.section} · стр. ${source.page_start} · ${source.chunk_id}`;
  }
  return link;
}

function renderSources(turn) {
  const box = element("div", "source-box");
  box.append(element("strong", "", "Источники"));
  if (turn.sources.length) {
    const list = element("div", "source-list");
    list.append(...turn.sources.map(sourceLink));
    box.append(list);
  } else {
    const message = turn.status === "done"
      ? "Нет подтверждённых источников."
      : "Поиск источников не завершён.";
    box.append(element("span", "no-source", message));
  }
  if (turn.citations.length) {
    const quotes = element("div", "quote-list");
    for (const citation of turn.citations) {
      quotes.append(element("blockquote", "", `«${citation.quote}»`));
    }
    box.append(quotes);
  }
  return box;
}

async function retry(turn) {
  if (busy || !currentSession) return;
  setBusy(true, "Повторяю ответ…");
  try {
    await api(`/api/rag-chat/sessions/${currentSession.id}/turns/${turn.id}/retry`, { method: "POST" });
    await openSession(currentSession.id);
    statusView.textContent = "Готово";
  } catch (error) {
    statusView.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

function configurationLabel(id) {
  return configurations.find((config) => config.id === id)?.label || id || "Прежние настройки";
}

function renderRuntime(profile, stats = {}, seconds = null) {
  const box = element("div", "runtime-metrics");
  if (profile) {
    const actual = stats.actual_options || {};
    box.append(element("span", "settings-line", `temperature ${profile.temperature} · контекст ${actual.num_ctx || profile.num_ctx} · ответ до ${actual.num_predict || profile.max_tokens} токенов`));
    box.append(element("span", "settings-line", `Prompt: ${profile.prompt_version === "baseline" ? "исходный" : "для конкурентного Java-кода"}`));
  }
  const values = [];
  if (seconds != null) values.push(`Ответ ${seconds.toFixed(2)} с`);
  if (stats.tokens_per_second != null) values.push(`${stats.tokens_per_second.toFixed(1)} токенов/с`);
  if (stats.output_tokens != null) values.push(`${stats.output_tokens} выходных токенов`);
  if (stats.load_seconds != null) values.push(`Загрузка ${stats.load_seconds.toFixed(2)} с`);
  box.append(element("span", "metric-line", values.join(" · ")));
  const allocation = stats.model_allocation;
  const modelBytes = allocation?.size ?? allocation?.size_vram;
  if (modelBytes != null) {
    box.append(element("strong", "allocation", `Память модели ${(modelBytes / 1e9).toFixed(2)} ГБ`));
    const quantization = allocation.details?.quantization_level;
    if (quantization) box.append(element("span", "settings-line", `Квантование ${quantization}`));
  }
  return box;
}

function renderComparison(turn) {
  const box = element("div", "comparison");
  const request = turn.metrics.comparison_request;
  box.append(element("p", "comparison-note", "Один вопрос, одинаковые найденные фрагменты и память. Качество сравнивайте по полноте ответа и цитатам."));
  const grid = element("div", "comparison-grid");
  for (const leg of ["reference", "candidate"]) {
    const config = request[leg];
    const result = turn.metrics.comparison?.[leg];
    const card = element("section", `comparison-card ${leg}`);
    card.append(element("h2", "", config.label));
    card.append(element("span", "comparison-model", config.model));
    if (result) {
      if (turn.metrics.comparison_reused?.includes(leg)) card.append(element("p", "comparison-note", "Сохранён с предыдущей попытки; повторная генерация не выполнялась."));
      card.append(element("p", "answer-text", result.answer.content));
      card.append(renderRuntime(config.profile, result.metrics, result.elapsed_seconds));
      if (!result.metrics.generation_attempts) card.append(element("p", "comparison-note", "Нет подходящих источников: ответ сформирован без генерации LLM."));
      const details = element("details", "comparison-sources");
      details.append(element("summary", "", `Источники и цитаты · ${result.answer.sources.length}`));
      details.append(renderSources({ ...result.answer, status: "done" }));
      card.append(details);
    } else {
      card.append(element("p", "answer-text", turn.status === "failed" ? "Ответ не завершён. Нажмите «Повторить ответ»." : "Готовлю ответ…"));
      card.append(renderRuntime(config.profile));
    }
    grid.append(card);
  }
  box.append(grid);
  const note = element("details", "comparison-measurement-note");
  note.append(element("summary", "", "Как измеряются время и память"));
  note.append(element("p", "comparison-note", "Память — выделение модели после ответа, без суммирования с RAM процесса. Время включает загрузку и перенастройку модели; повторные запросы могут быть быстрее."));
  box.append(note);
  return box;
}

function renderMessages() {
  messagesView.replaceChildren();
  if (!currentSession?.turns.length) {
    const empty = element("div", "empty");
    empty.append(element("span", "empty-mark", "✦"));
    empty.append(element("h2", "", "Спросите по книге"));
    empty.append(element("p", "", "Сформулируйте цель, уточните ограничения или сразу задайте вопрос. История, память задачи и источники сохраняются в этом чате."));
    messagesView.append(empty);
    return;
  }
  for (const turn of currentSession.turns) {
    const pair = element("div", "turn");
    const user = element("article", "bubble user");
    user.id = `turn-${turn.id}`;
    user.append(element("span", "bubble-label", `ВЫ · ${turn.position}`));
    user.append(element("p", "answer-text", turn.content));
    pair.append(user);

    const assistant = element("article", "bubble assistant");
    assistant.append(element("span", "bubble-label", modelLabel(turn.provider, turn.model)
      + (turn.metrics?.generation_profile ? ` · ${configurationLabel(turn.metrics.configuration_id || turn.metrics.generation_profile.name)}` : "")));
    const message = turn.status === "failed"
      ? turn.error || "Не удалось ответить."
      : turn.status === "pending" ? "Готовлю ответ…" : turn.answer;
    const comparing = turn.metrics?.comparison_request && !turn.metrics?.comparison_note;
    if (comparing) {
      assistant.classList.add("comparison-bubble");
      assistant.append(renderComparison(turn));
      if (turn.error) assistant.append(element("p", "comparison-error", turn.error));
    } else {
      assistant.append(element("p", "answer-text", message));
      if (turn.metrics?.generation_profile) assistant.append(renderRuntime(
        turn.metrics.generation_profile, turn.metrics.generation_statistics,
        turn.metrics.generation_seconds,
      ));
      if (turn.metrics?.comparison_note) assistant.append(element("p", "comparison-note", turn.metrics.comparison_note));
      assistant.append(renderSources(turn));
    }
    if (turn.metrics?.total_seconds != null) {
      const m = turn.metrics;
      assistant.append(element("p", "turn-timing",
        `Всего ${m.total_seconds.toFixed(1)} с · поиск ${(m.retrieval_seconds || 0).toFixed(1)} с`
        + (comparing ? ` · общий контекст для двух вариантов` : ` · ответ ${(m.generation_seconds || 0).toFixed(1)} с`)));
    }
    if (turn.status !== "done" && turn === currentSession.turns.at(-1)) {
      const button = element("button", "retry-button", "Повторить ответ");
      button.type = "button";
      button.disabled = busy;
      button.addEventListener("click", () => retry(turn));
      assistant.append(button);
    }
    pair.append(assistant);
    messagesView.append(pair);
  }
  const latest = currentSession.turns.at(-1);
  if (latest?.metrics?.comparison_request && messagesView.lastElementChild) {
    messagesView.lastElementChild.scrollIntoView({ block: "start" });
  } else messagesView.scrollTop = messagesView.scrollHeight;
}

function renderFact(target, fact) {
  const container = document.querySelector(target);
  container.replaceChildren();
  const facts = Array.isArray(fact) ? fact : fact ? [fact] : [];
  if (!facts.length) {
    container.append(element("p", "state-empty", "Пока не зафиксировано"));
    return;
  }
  for (const item of facts) {
    const card = element("div", "fact");
    if (item.key !== "goal") card.append(element("strong", "", item.key));
    card.append(element("p", "", item.value));
    card.append(sourceLink({
      kind: "message", message_id: item.source_message_id,
      title: "Показать исходное сообщение →",
    }));
    container.append(card);
  }
}

function renderState() {
  const state = currentSession?.state || {
    goal: null, constraints: [], terms: [], clarifications: [], revision: 0,
  };
  document.querySelector("#state-revision").textContent = `v${state.revision}`;
  renderFact("#state-goal", state.goal);
  renderFact("#state-constraints", state.constraints);
  renderFact("#state-terms", state.terms);
  renderFact("#state-clarifications", state.clarifications);
}

async function refreshSessions() {
  const sessions = await api("/api/rag-chat/sessions");
  sessionsView.replaceChildren();
  for (const session of sessions) {
    const button = element("button", `session-link${currentSession?.id === session.id ? " active" : ""}`, session.title);
    button.type = "button";
    button.addEventListener("click", () => { if (!busy) void openSession(session.id); });
    sessionsView.append(button);
  }
  return sessions;
}

async function openSession(id) {
  currentSession = await api(`/api/rag-chat/sessions/${id}`);
  referenceConfiguration = currentSession.turns.at(-1)?.metrics?.comparison_request?.reference?.configuration_id || "baseline";
  localStorage.setItem("ragChatSessionId", id);
  chatTitle.textContent = currentSession.title;
  renderMessages();
  renderModel();
  renderState();
  showMemory(!currentSession.turns.at(-1)?.metrics?.comparison_request);
  updateComposer();
  await refreshSessions();
}

async function createSession() {
  const session = await api("/api/rag-chat/sessions", { method: "POST" });
  await openSession(session.id);
  input.focus();
}

function setBusy(value, statusText = null) {
  busy = value;
  newButton.disabled = value;
  if (statusText) statusView.textContent = statusText;
  if (timer) clearInterval(timer);
  if (value) {
    const started = Date.now();
    timer = setInterval(async () => {
      statusView.textContent = `${statusText || "Готовлю ответ…"} ${Math.floor((Date.now() - started) / 1000)} с`;
      if (!currentSession || polling) return;
      const id = currentSession.id;
      polling = true;
      try {
        const session = await api(`/api/rag-chat/sessions/${id}`);
        if (busy && currentSession?.id === id && session.turns.length
            && JSON.stringify(currentSession.turns) !== JSON.stringify(session.turns)) {
          currentSession.turns = session.turns;
          currentSession.state = session.state;
          renderMessages();
          renderState();
        }
      } catch { /* The send/retry response reports actionable failures. */ }
      finally { polling = false; }
    }, 1000);
  }
  document.querySelectorAll(".retry-button").forEach((button) => { button.disabled = value; });
  updateComposer();
}

function updateComposer() {
  const unresolved = currentSession?.turns.at(-1)?.status !== "done"
    && Boolean(currentSession?.turns.length);
  input.disabled = busy || unresolved;
  sendButton.disabled = busy || unresolved;
  comparisonSelect.disabled = busy || currentSession?.provider !== "ollama";
  compareButton.disabled = busy || unresolved || currentSession?.provider !== "ollama"
    || !configurations.find((c) => c.id === selectedConfiguration())?.available
    || !configurations.find((c) => c.id === referenceConfiguration)?.available;
  if (!busy && unresolved) {
    statusView.textContent = "Повторите последний ответ перед новым сообщением";
  }
}

async function send(content, compare = false) {
  if (busy) return;
  setBusy(true, compare ? "Сравниваю два варианта…" : "Ищу контекст и готовлю ответ…");
  try {
    if (!currentSession) await createSession();
    const sessionId = currentSession.id;
    const provider = currentSession.provider;
    const configuration = selectedConfiguration();
    const compareWith = compare ? referenceConfiguration : null;
    if (compare) showMemory(false);
    currentSession.turns.push({ id: "pending-ui", position: currentSession.turns.length + 1,
      content, answer: "", status: "pending", sources: [], citations: [], provider,
      model: provider === "ollama" ? configurations.find((c) => c.id === configuration)?.model
        : models.find((m) => m.id === provider)?.model, metrics: {} });
    renderMessages();
    input.value = "";
    await api(`/api/rag-chat/sessions/${sessionId}/turns`, {
      method: "POST", body: JSON.stringify({ content, provider,
        configuration: provider === "ollama" ? configuration : null, compare_with: compareWith }),
    });
    statusView.textContent = "Готово";
    input.value = "";
    await openSession(currentSession.id);
  } catch (error) {
    try {
      await openSession(currentSession.id);
    } catch {
      currentSession.turns = currentSession.turns.filter((turn) => turn.id !== "pending-ui");
      input.value = content;
      renderMessages();
    }
    statusView.textContent = error.message;
  } finally {
    setBusy(false);
    input.focus();
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const content = input.value.trim();
  if (content) void send(content);
});
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    composer.requestSubmit();
  }
});
newButton.addEventListener("click", () => {
  if (!busy) void createSession().catch((error) => { statusView.textContent = error.message; });
});
compareButton.addEventListener("click", () => {
  const content = input.value.trim();
  if (content) void send(content, true);
  else input.focus();
});

async function initialize() {
  try {
    const health = await api("/api/health");
    if (health.network_mode === "loopback_only") {
      document.querySelector(".header-controls").append(
        element("span", "offline-badge", "Офлайн · только локальные соединения"));
    }
    models = (await api("/api/models")).providers;
    const catalog = await api("/api/rag-chat/configurations");
    configurations = catalog.configurations;
    defaultConfiguration = catalog.default;
    const sessions = await refreshSessions();
    const requested = new URLSearchParams(window.location.search).get("session");
    const saved = localStorage.getItem("ragChatSessionId");
    const selected = sessions.find((session) => session.id === requested)
      || sessions.find((session) => session.id === saved) || sessions[0];
    if (selected) await openSession(selected.id);
    else await createSession();
    const index = await api("/api/document-index/status");
    document.querySelector("#index-status").textContent = index.state === "ready"
      ? `${index.index.pages} стр. · индекс готов`
      : "Подготовьте локальный индекс";
  } catch (error) {
    statusView.textContent = error.message;
  }
}

void initialize();
