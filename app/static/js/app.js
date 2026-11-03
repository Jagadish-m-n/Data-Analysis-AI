/* Data Analysis AI - IDE-style browser UI: files | file viewer | chat (same-origin backend).
 *
 *   GET    /api/status                       -> status badge and limits
 *   POST   /api/threads                      -> upload CSV files: starts a conversation (sandbox + agent)
 *   GET    /api/threads/{id}                 -> restores the conversation after a page reload
 *   POST   /api/threads/{id}/messages        -> ask; the agent's steps stream back live (Server-Sent Events)
 *   POST   /api/threads/{id}/stop            -> Stop button
 *   DELETE /api/threads/{id}                 -> New conversation
 *   GET    /api/threads/{id}/tree, /file     -> file tree and file viewer
 *
 * Files sync while the agent works: after write_file / edit_file that file is reloaded, after
 * execute / task (which can change anything) the whole tree is. Before each question the files
 * are remembered, so the ones the agent changes are marked and open in a diff view.
 *
 * Everything that comes from data files, the sandbox or the model is inserted as text, or as
 * markdown cleaned by DOMPurify - never as raw HTML.
 */
"use strict";

const THREAD_KEY = "daa.threadId"; // sessionStorage: a reload reconnects to the same sandbox
const KEY_STORAGE = "daa.apiKey"; // sessionStorage: forgotten when the tab closes
const THEME_STORAGE = "daa.theme";
// Used until /api/status answers (same defaults as app/config/settings.py)
const DEFAULT_LIMITS = { maxFiles: 10, maxMb: 50, maxChars: 2000 };
const MAX_VIEW_BYTES = 5000000; // same limit as the backend's file route
const SNAPSHOT_MAX_BYTES = 200000; // text files up to this size are remembered before each question, for diffs
const CSV_PREVIEW_ROWS = 200;
const DIFF_CONTEXT = 3; // unchanged lines shown around each change
const TREE_ROOTS = ["/data", "/output", "/skills"];
const SKILLS_ROOT = "/skills";
const IMAGE_TYPES = { png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", gif: "image/gif", webp: "image/webp", svg: "image/svg+xml" };
const LANGUAGES = { py: "python", md: "markdown", json: "json", sh: "bash", yaml: "yaml", yml: "yaml", toml: "ini", sql: "sql", html: "xml", xml: "xml", js: "javascript", r: "r" };
const FILE_ICONS = { csv: "filetype-csv", tsv: "filetype-csv", md: "filetype-md", py: "filetype-py", png: "filetype-png", jpg: "filetype-jpg", svg: "filetype-svg", json: "filetype-json", txt: "filetype-txt", sh: "filetype-sh" };
const MODE_LABELS = { diff: "Diff", preview: "Preview", table: "Table", code: "Code" };

const state = {
  status: null, // last /api/status response
  ready: false, // backend configured well enough to run the agent
  threadId: null,
  sandboxId: null,
  dataFiles: [],
  pending: [], // File objects chosen before the conversation starts
  tree: new Map(), // path -> {path, type: "file" | "dir", size}
  touched: new Set(), // files the agent wrote outside TREE_ROOTS - shown under "Other files"
  collapsed: new Set([SKILLS_ROOT]), // folders the user closed in the tree
  contents: new Map(), // path -> {kind: "text" | "image" | "binary" | "toolarge", text?, url?, bytes?, size}
  // What the files looked like when the last question was sent
  snapshotTaken: false,
  beforeSizes: new Map(), // path -> size
  before: new Map(), // path -> text (small text files only)
  written: new Set(), // paths write_file / edit_file touched since then
  changed: new Set(),
  diffCache: new Map(), // path -> {before, text, parts}
  selected: null, // path open in the viewer
  mode: null, // its view: "diff" | "preview" | "table" | "code" (null = the default for the file)
  run: null, // {controller, calls: Map tool_call id -> event} while a question is answered
};

const $ = (selector) => document.querySelector(selector);

// ======================================================================
// DOM helpers
// ======================================================================
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child != null && child !== false) node.append(child);
  }
  return node;
}

const icon = (name, extra = "") => el("i", { class: `bi bi-${name} ${extra}`.trim(), "aria-hidden": "true" });

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

const baseName = (path) => path.slice(path.lastIndexOf("/") + 1);
const dirName = (path) => path.slice(0, path.lastIndexOf("/")) || "/";
const extOf = (path) => {
  const name = baseName(path);
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : "";
};
const isImage = (path) => extOf(path) in IMAGE_TYPES;
const isSkill = (path) => path.startsWith(SKILLS_ROOT + "/");
const rootOf = (path) => TREE_ROOTS.find((root) => path === root || path.startsWith(root + "/"));
const fileIcon = (path) => FILE_ICONS[extOf(path)] || "file-earmark";

function limits() {
  const s = state.status;
  if (!s) return DEFAULT_LIMITS;
  return { maxFiles: s.sandbox.max_upload_files, maxMb: s.sandbox.max_upload_mb, maxChars: s.agent.max_question_chars };
}

function alertBox(kind, message) {
  return el(
    "div",
    { class: `alert alert-${kind} alert-dismissible fade show py-2 small mb-0`, role: "alert" },
    message,
    el("button", { type: "button", class: "btn-close py-2", "data-bs-dismiss": "alert", "aria-label": "Close" })
  );
}

function showNotice(kind, message) {
  $("#messages").append(alertBox(kind, message));
  scrollChat();
}

function scrollChat() {
  const box = $("#messages");
  box.scrollTop = box.scrollHeight;
}

// Markdown images with no scheme (chart.png, /output/chart.png) are files in the sandbox: their
// src is moved aside before the browser can request them from this server, and loadSandboxImages
// fetches them through the file route instead
const HAS_SCHEME = /^[a-z][a-z0-9+.-]*:/i;
if (window.DOMPurify) {
  DOMPurify.addHook("afterSanitizeAttributes", (node) => {
    const src = node.tagName === "IMG" ? node.getAttribute("src") : null;
    if (src && !HAS_SCHEME.test(src)) {
      node.setAttribute("data-sandbox-src", src);
      node.removeAttribute("src");
    }
  });
}

/** Markdown -> sanitized HTML. Relative image paths are resolved against baseDir in the sandbox. */
function renderMarkdown(markdown, className, baseDir = "/output") {
  const box = el("div", { class: className });
  if (window.marked && window.DOMPurify) {
    box.innerHTML = DOMPurify.sanitize(marked.parse(markdown || ""));
    box.querySelectorAll("table").forEach((t) => t.classList.add("table", "table-sm", "table-bordered"));
    box.querySelectorAll("a").forEach((a) => {
      a.target = "_blank";
      a.rel = "noopener noreferrer";
    });
    loadSandboxImages(box, baseDir);
  } else {
    // CDN blocked: show the plain text rather than risk unsanitized HTML
    box.style.whiteSpace = "pre-wrap";
    box.textContent = markdown || "";
  }
  return box;
}

function loadSandboxImages(container, baseDir) {
  container.querySelectorAll("img[data-sandbox-src]").forEach((img) => {
    const path = resolvePath(baseDir, img.dataset.sandboxSrc);
    img.title = path;
    if (!state.threadId) return;
    ensureContent(path)
      .then((content) => {
        if (content.kind === "image") img.src = content.url;
      })
      .catch(() => (img.alt = `${img.alt || path} (not found in the sandbox)`));
  });
}

/** "chart.png" in /output -> "/output/chart.png"; also handles "./", "../" and absolute paths. */
function resolvePath(baseDir, relative) {
  const parts = [];
  for (const part of (relative.startsWith("/") ? relative : `${baseDir}/${relative}`).split("/")) {
    if (part === "..") parts.pop();
    else if (part && part !== ".") parts.push(part);
  }
  return "/" + parts.join("/");
}

// ======================================================================
// API
// ======================================================================
class ApiError extends Error {
  constructor(message, code, status) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

function authHeaders() {
  const key = sessionStorage.getItem(KEY_STORAGE);
  return key ? { "X-API-Key": key } : {};
}

/** fetch() that throws ApiError with the backend's {"error": {...}} message when it fails. */
async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, { ...options, headers: { ...authHeaders(), ...options.headers } });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError("Cannot reach the backend. Is the server running?", "NETWORK_ERROR", 0);
  }
  if (!response.ok) {
    let error = null;
    try {
      error = (await response.json()).error;
    } catch {
      // not JSON - handled below
    }
    throw new ApiError(
      error ? error.message : `The server returned HTTP ${response.status}.`,
      error ? error.code : `HTTP_${response.status}`,
      response.status
    );
  }
  return response;
}

async function api(path, options) {
  const response = await request(path, options);
  try {
    return await response.json();
  } catch {
    throw new ApiError("The backend returned a response that is not JSON.", "BAD_RESPONSE", response.status);
  }
}

/** POST a JSON body and call onEvent for each Server-Sent Event ("data: {...}") in the reply. */
async function streamEvents(path, body, signal, onEvent) {
  const response = await request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const data = buffer
        .slice(0, end)
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      buffer = buffer.slice(end + 2);
      if (data) onEvent(JSON.parse(data));
    }
  }
}

const threadUrl = (suffix = "") => `/api/threads/${encodeURIComponent(state.threadId)}${suffix}`;
const errorText = (err) => (err.code && err.code !== "NETWORK_ERROR" ? `${err.message} (${err.code})` : err.message);

// ======================================================================
// Status, API key and theme
// ======================================================================
function setBadge(kind, text, title = "") {
  const badge = $("#status-badge");
  badge.className = `status-pill status-${kind}`;
  badge.textContent = text;
  badge.title = title;
}

async function loadStatus() {
  setBadge("secondary", "Checking…");
  try {
    state.status = await api("/api/status");
  } catch (err) {
    state.status = null;
    state.ready = false;
    if (err.status === 401) {
      setBadge("warning", "API key needed");
      showKeyPrompt();
    } else {
      setBadge("danger", "Offline");
      showNotice("danger", errorText(err));
    }
    updateControls();
    return false;
  }

  const { agent, sandbox } = state.status;
  const problems = [];
  if (!agent.enabled) problems.push("The AI agent is disabled (ENABLE_AI_AGENT=false).");
  if (!agent.api_key_set) problems.push(`${(agent.provider || "LLM").toUpperCase()}_API_KEY is not set in .env.`);
  if (!sandbox.langsmith_api_key_set) problems.push("LANGSMITH_API_KEY is not set in .env.");
  state.ready = problems.length === 0;
  setBadge(state.ready ? "success" : "warning", state.ready ? "Online" : "Not ready", `${agent.provider} · ${agent.model}`);
  const model = [agent.provider, agent.model].filter(Boolean).join(" · ");
  $("#model-label").replaceChildren(...(model ? [icon("cpu", "me-1"), model] : []));
  $("#model-label").title = model && `Model: ${model}`;
  problems.forEach((problem) => showNotice("warning", problem));

  $("#question").maxLength = limits().maxChars;
  updateCounter();
  updateFileHint();
  updateControls();
  return true;
}

function showKeyPrompt() {
  const input = el("input", { type: "password", class: "form-control form-control-sm", placeholder: "API key", autocomplete: "off", "aria-label": "API key" });
  const form = el(
    "form",
    { class: "alert alert-warning small mb-0 d-flex flex-column gap-2" },
    el("div", { text: "This server requires an API key (API_AUTH_KEY). It is kept only until you close this tab." }),
    el("div", { class: "d-flex gap-2" }, input, el("button", { type: "submit", class: "btn btn-warning btn-sm" }, "Save"))
  );
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!input.value.trim()) return;
    sessionStorage.setItem(KEY_STORAGE, input.value.trim());
    form.remove();
    if (await loadStatus()) await restoreThread();
  });
  $("#messages").append(form);
  input.focus();
}

function applyTheme(theme) {
  document.documentElement.dataset.bsTheme = theme;
  $("#hljs-light").disabled = theme === "dark";
  $("#hljs-dark").disabled = theme !== "dark";
  $("#theme-toggle").replaceChildren(icon(theme === "dark" ? "sun" : "moon-stars"));
}

// ======================================================================
// Conversation lifecycle
// ======================================================================
function setThread(thread) {
  state.threadId = thread.thread_id;
  state.sandboxId = thread.sandbox_id;
  state.dataFiles = thread.data_files;
  sessionStorage.setItem(THREAD_KEY, thread.thread_id);
  updateControls();
}

/** Forget the conversation in the browser (the sandbox itself is deleted by the caller). */
function forgetThread() {
  sessionStorage.removeItem(THREAD_KEY);
  for (const content of state.contents.values()) if (content.url) URL.revokeObjectURL(content.url);
  Object.assign(state, {
    threadId: null,
    sandboxId: null,
    dataFiles: [],
    tree: new Map(),
    touched: new Set(),
    contents: new Map(),
    snapshotTaken: false,
    beforeSizes: new Map(),
    before: new Map(),
    written: new Set(),
    changed: new Set(),
    diffCache: new Map(),
    selected: null,
    mode: null,
  });
  renderTree();
  renderChanges();
  renderViewerEmpty();
  updateControls();
}

/** After a page reload: reconnect to the conversation (and sandbox) this tab had open. */
async function restoreThread() {
  const id = sessionStorage.getItem(THREAD_KEY);
  if (!id || state.threadId) return;
  let thread;
  try {
    thread = await api(`/api/threads/${encodeURIComponent(id)}`);
  } catch (err) {
    if (err.status === 404) sessionStorage.removeItem(THREAD_KEY);
    else showNotice("warning", `Could not reconnect to the conversation: ${errorText(err)}`);
    return;
  }
  setThread(thread);
  $("#messages").replaceChildren();
  let turn = null;
  for (const event of thread.events) {
    if (event.type === "user") {
      if (turn) turn.finish();
      turn = startTurn(event.text);
    } else if (turn) renderEvent(turn, event);
  }
  if (turn) turn.finish();
  if (thread.busy) showNotice("info", "The agent is still answering the question sent before the page was reloaded. Refresh the files to see its work.");
  await syncFiles({ tree: true });
}

async function newThread() {
  if (state.run) state.run.controller.abort();
  const id = state.threadId;
  forgetThread();
  state.pending = [];
  renderPendingFiles();
  welcome();
  if (!id) return;
  try {
    await api(`/api/threads/${encodeURIComponent(id)}`, { method: "DELETE" });
  } catch (err) {
    if (err.status !== 404) showNotice("warning", `The previous sandbox could not be deleted: ${errorText(err)}`);
  }
}

// ======================================================================
// Asking a question
// ======================================================================
async function send(event) {
  event.preventDefault();
  if (state.run || !state.ready) return;
  const question = $("#question").value.trim();
  if (!state.threadId && state.pending.length === 0) return showNotice("warning", "Add at least one CSV file first.");
  if (!question) return $("#question").focus();

  const run = { controller: new AbortController(), calls: new Map(), ended: false };
  state.run = run;
  $("#question").value = "";
  updateCounter();
  updateControls();
  const turn = startTurn(question);

  try {
    if (!state.threadId) await startThread(turn, run);
    await snapshot();
    if (run.stopped) throw new DOMException("Stopped before the question was sent", "AbortError");
    await streamEvents(threadUrl("/messages"), { question }, run.controller.signal, (e) => handleLiveEvent(turn, run, e));
    if (!run.ended) addToTurn(turn, errorNote("The connection to the server closed before the agent finished."));
  } catch (err) {
    if (err.name === "AbortError") renderEvent(turn, { type: "stopped" });
    else {
      addToTurn(turn, errorNote(errorText(err)));
      if (!state.threadId) $("#question").value = question; // nothing was sent: keep it for another try
      if (err.code === "THREAD_NOT_FOUND") {
        forgetThread();
        showNotice("info", "This conversation expired and its sandbox is gone. Add your CSV files again to start a new one.");
      }
    }
  } finally {
    turn.finish();
    state.run = null;
    updateCounter();
    updateControls();
    if (state.threadId) await syncFiles({ tree: true });
  }
}

async function startThread(turn, run) {
  const note = addToTurn(
    turn,
    el("div", { class: "msg-note" }, el("span", { class: "spinner-border spinner-border-sm me-2", "aria-hidden": "true" }), `Starting a sandbox with ${state.pending.length} file(s)…`)
  );
  const form = new FormData();
  state.pending.forEach((file) => form.append("files", file, file.name));
  const thread = await api("/api/threads", { method: "POST", body: form, signal: run.controller.signal });
  note.remove();
  state.pending = [];
  renderPendingFiles();
  setThread(thread);
}

async function stop() {
  const run = state.run;
  if (!run) return;
  run.stopped = true; // a question that hasn't been sent yet is then not sent at all
  if (!state.threadId) return run.controller.abort(); // still creating the sandbox
  try {
    const { status } = await api(threadUrl("/stop"), { method: "POST" }); // the stream then ends with "stopped"
    if (status === "not_running") run.controller.abort(); // closing the stream stops a run that is just starting
  } catch {
    run.controller.abort();
  }
}

function handleLiveEvent(turn, run, event) {
  renderEvent(turn, event);
  if (["answer", "error", "stopped"].includes(event.type)) run.ended = true;
  if (event.type === "tool_call" && event.id) run.calls.set(event.id, event);
  if (event.type !== "tool_result") return;

  const call = run.calls.get(event.id);
  if (!call) return;
  const path = call.args.file_path || call.args.path;
  if ((call.name === "write_file" || call.name === "edit_file") && path) {
    state.written.add(path);
    if (!rootOf(path)) state.touched.add(path);
    syncFiles({ tree: true, reload: [path] });
  } else if (call.name === "execute" || call.name === "task") {
    syncFiles({ tree: true }); // a script or the subagent can change any file
  }
}

// ======================================================================
// Chat rendering
// ======================================================================
// Example questions on the welcome screen; a click puts one in the question box
const SUGGESTIONS = [
  { icon: "magic", text: "Find data-quality issues and clean the data" },
  { icon: "graph-up-arrow", text: "What are the main trends? Show them in a chart." },
  { icon: "file-earmark-bar-graph", text: "Summarize the key KPIs in a short report" },
];

function welcome() {
  const suggestion = ({ icon: name, text }) =>
    el("button", { type: "button", class: "suggestion", onclick: () => useSuggestion(text) }, icon(name), el("span", { text }), icon("arrow-right", "arrow"));
  $("#messages").replaceChildren(
    el(
      "div",
      { class: "welcome" },
      el("div", { class: "welcome-mark", "aria-hidden": "true" }, icon("bar-chart-line")),
      el("h2", { class: "welcome-title", text: "Analyze your data in plain English" }),
      el(
        "p",
        { class: "welcome-text" },
        "Upload CSV files and ask a question. The agent cleans, analyzes and charts the data in an isolated sandbox: your files are in ",
        el("code", { text: "/data" }),
        ", its charts and reports appear in ",
        el("code", { text: "/output" }),
        "."
      ),
      el("div", { class: "suggestions" }, el("div", { class: "suggestions-label", text: "Try asking" }), SUGGESTIONS.map(suggestion)),
      el(
        "div",
        { class: "features" },
        el("span", {}, icon("shield-check"), "Isolated sandbox"),
        el("span", {}, icon("activity"), "Live step-by-step"),
        el("span", {}, icon("chat-left-dots"), "Follow-up questions")
      )
    )
  );
}

function useSuggestion(text) {
  const question = $("#question");
  question.value = text;
  updateCounter();
  question.focus();
}

function startTurn(question) {
  $("#messages .welcome")?.remove();
  const body = el("div", { class: "turn" });
  $("#messages").append(el("div", { class: "msg-user", text: question }), body);
  scrollChat();
  return {
    body,
    list: null, // the <ul> new steps go into
    steps: new Map(), // tool_call id -> <li>
    finish() {
      for (const li of this.steps.values()) if (li.dataset.pending) markStep(li, "stopped");
    },
  };
}

function addToTurn(turn, node) {
  turn.body.append(node);
  turn.list = null; // steps after this node start a new list
  scrollChat();
  return node;
}

function errorNote(message) {
  return el("div", { class: "alert alert-danger py-2 small mb-0", role: "alert", text: message });
}

function renderEvent(turn, event) {
  switch (event.type) {
    case "note":
      addToTurn(turn, el("div", { class: "msg-note", text: event.text }));
      break;
    case "tool_call": {
      const li = el(
        "li",
        { class: `step${event.agent ? " sub" : ""}`, "data-pending": "1" },
        el(
          "div",
          { class: "step-line" },
          el("span", { class: "spinner-border text-secondary", style: "width: .75rem; height: .75rem; border-width: .12rem", "aria-hidden": "true" }),
          el("span", {}, event.agent && el("span", { class: "badge text-bg-secondary me-1", text: event.agent }), describeStep(event.name, event.args || {}))
        )
      );
      if (!turn.list) turn.list = turn.body.appendChild(el("ul", { class: "steps" }));
      turn.list.append(li);
      if (event.id) turn.steps.set(event.id, li);
      scrollChat();
      break;
    }
    case "tool_result": {
      const li = turn.steps.get(event.id);
      if (!li) break;
      markStep(li, event.ok ? "ok" : "failed");
      if (event.output && (!event.ok || event.name === "execute")) {
        const output = el("pre", { class: "step-output", text: event.output });
        li.append(event.ok ? el("details", { class: "step-details" }, el("summary", { text: "output" }), output) : output);
      }
      break;
    }
    case "answer":
      addToTurn(turn, el("div", { class: "msg-agent" }, el("span", { class: "agent-avatar", "aria-hidden": "true" }, icon("stars")), renderMarkdown(event.text, "msg-answer")));
      break;
    case "error":
      addToTurn(turn, errorNote(`${event.message} (${event.code})`));
      break;
    case "stopped":
      addToTurn(turn, el("div", { class: "msg-note fst-italic" }, icon("stop-circle", "me-1"), "Stopped."));
      break;
  }
}

function markStep(li, result) {
  delete li.dataset.pending;
  const name = { ok: "check2 text-success", failed: "x-circle text-danger", stopped: "dash-circle text-secondary" }[result];
  li.querySelector(".step-line").firstElementChild.replaceWith(el("i", { class: `bi bi-${name}`, title: result }));
}

/** One readable line per tool call, e.g. "Ran `python /output/analyze.py`". */
function describeStep(name, args) {
  const path = args.file_path || args.path || "";
  switch (name) {
    case "execute":
      return ["Ran ", el("code", { text: args.command || "" })];
    case "write_file":
      return ["Wrote ", fileLink(path)];
    case "edit_file":
      return ["Edited ", fileLink(path)];
    case "read_file":
      if (isSkill(path)) return ["Read the ", el("code", { text: path.split("/")[2] }), " skill"];
      return ["Read ", fileLink(path)];
    case "ls":
      return ["Listed ", el("code", { text: path || "/" })];
    case "task":
      return ["Asked the ", el("code", { text: args.subagent_type || "subagent" }), ` subagent: ${args.description || ""}`];
    default:
      return ["Called ", el("code", { text: name })];
  }
}

const fileLink = (path) => el("button", { type: "button", class: "step-link", text: path, onclick: () => openFile(path) });

// ======================================================================
// Files: tree, snapshots and changes
// ======================================================================
let syncChain = Promise.resolve();

/** Refresh the tree and/or files. Calls are queued, so overlapping refreshes can't interleave. */
function syncFiles({ tree = false, reload = [] } = {}) {
  syncChain = syncChain.then(async () => {
    if (!state.threadId) return;
    try {
      if (tree) await refreshTree();
      reload.forEach((path) => state.contents.delete(path));
      computeChanges();
      // Changed text files are loaded for the "+added -removed" counts
      await Promise.all(
        [...state.changed]
          .filter((path) => state.tree.has(path) && !isImage(path) && (state.tree.get(path).size ?? 0) <= SNAPSHOT_MAX_BYTES)
          .map((path) => ensureContent(path).catch(() => null))
      );
      renderTree();
      renderChanges();
      const selected = state.selected;
      if (selected && (reload.includes(selected) || isStale(selected) || state.changed.has(selected))) {
        await openFile(selected, { keepMode: true });
      }
    } catch (err) {
      console.warn("File sync failed:", err);
      if (err.code === "THREAD_NOT_FOUND") forgetThread();
    }
  });
  return syncChain;
}

async function refreshTree() {
  const { entries } = await api(threadUrl("/tree"));
  const tree = new Map(entries.map((entry) => [entry.path, entry]));
  for (const path of state.touched) {
    if (!tree.has(path)) tree.set(path, { path, type: "file", size: state.contents.has(path) ? state.contents.get(path).size : null });
  }
  state.tree = tree;
}

/** Remember the files as they are now, so the agent's changes to them can be shown. */
async function snapshot() {
  await refreshTree();
  state.beforeSizes = new Map();
  state.before = new Map();
  state.written = new Set();
  state.diffCache = new Map();
  state.snapshotTaken = true;
  const loads = [];
  for (const [path, entry] of state.tree) {
    if (entry.type !== "file" || isSkill(path)) continue;
    state.beforeSizes.set(path, entry.size);
    if (!isImage(path) && entry.size != null && entry.size <= SNAPSHOT_MAX_BYTES) {
      loads.push(
        ensureContent(path)
          .then((content) => content.kind === "text" && state.before.set(path, content.text))
          .catch(() => null)
      );
    }
  }
  await Promise.all(loads);
  computeChanges();
  renderTree();
  renderChanges();
}

function computeChanges() {
  state.changed = new Set();
  if (!state.snapshotTaken) return;
  for (const [path, entry] of state.tree) {
    if (entry.type !== "file" || isSkill(path)) continue;
    const before = state.beforeSizes.get(path);
    if (before === undefined || before !== entry.size || state.written.has(path)) state.changed.add(path);
  }
  for (const path of state.beforeSizes.keys()) if (!state.tree.has(path)) state.changed.add(path); // deleted
}

function changeKind(path) {
  if (!state.changed.has(path)) return null;
  if (!state.beforeSizes.has(path)) return "added";
  if (!state.tree.has(path)) return "deleted";
  return "modified";
}

/** jsdiff line diff of a changed text file against the snapshot, or null when there is none. */
function diffParts(path) {
  if (!window.Diff || !state.changed.has(path)) return null;
  const existed = state.beforeSizes.has(path);
  if (existed && !state.before.has(path)) return null; // was too large to remember
  const current = state.tree.has(path) ? state.contents.get(path) : { kind: "text", text: "" };
  if (!current || current.kind !== "text") return null;
  const before = existed ? state.before.get(path) : "";
  const cached = state.diffCache.get(path);
  if (cached && cached.before === before && cached.text === current.text) return cached.parts;
  const parts = Diff.diffLines(before, current.text);
  state.diffCache.set(path, { before, text: current.text, parts });
  return parts;
}

function diffStats(path) {
  const parts = diffParts(path);
  if (!parts) return null;
  let added = 0;
  let removed = 0;
  for (const part of parts) {
    if (part.added) added += part.count;
    else if (part.removed) removed += part.count;
  }
  return { added, removed };
}

const isStale = (path) => {
  const cached = state.contents.get(path);
  const entry = state.tree.get(path);
  return Boolean(cached && entry && entry.size != null && cached.size !== entry.size);
};

/** The content of a file, from the cache or the sandbox. */
async function ensureContent(path) {
  const entry = state.tree.get(path);
  const cached = state.contents.get(path);
  if (cached && !isStale(path)) return cached;
  if (entry && entry.size > MAX_VIEW_BYTES) return setContent(path, { kind: "toolarge", size: entry.size });

  const response = await request(threadUrl(`/file?path=${encodeURIComponent(path)}`));
  const bytes = new Uint8Array(await response.arrayBuffer());
  let content;
  if (isImage(path)) {
    // A blob: URL in an <img> can't run scripts, even for an SVG the agent wrote
    content = { kind: "image", url: URL.createObjectURL(new Blob([bytes], { type: IMAGE_TYPES[extOf(path)] })) };
  } else if (bytes.subarray(0, 8192).includes(0)) {
    content = { kind: "binary" };
  } else {
    content = { kind: "text", text: new TextDecoder().decode(bytes) };
  }
  content.bytes = bytes;
  content.size = bytes.length;
  return setContent(path, content);
}

function setContent(path, content) {
  const old = state.contents.get(path);
  if (old && old.url) URL.revokeObjectURL(old.url);
  state.contents.set(path, content);
  return content;
}

// ======================================================================
// Files: rendering
// ======================================================================
const indent = (depth) => `padding-left: ${0.5 + depth * 0.85}rem`;

function renderTree() {
  const box = $("#tree");
  if (!state.threadId) {
    box.replaceChildren(
      el("div", { class: "empty-note" }, "The sandbox files appear here once the conversation starts: your data in ", el("code", { text: "/data" }), ", the agent's charts and reports in ", el("code", { text: "/output" }), ".")
    );
    return;
  }

  const children = new Map(); // folder -> its entries
  for (const entry of state.tree.values()) {
    if (!rootOf(entry.path)) continue;
    const parent = dirName(entry.path);
    if (!children.has(parent)) children.set(parent, []);
    children.get(parent).push(entry);
  }
  const sorted = (entries) =>
    entries.sort((a, b) => (a.type === b.type ? a.path.localeCompare(b.path) : a.type === "dir" ? -1 : 1));

  const folder = (path, label, depth, entries) => {
    const details = el(
      "details",
      { open: !state.collapsed.has(path) },
      el(
        "summary",
        { class: "tree-row", style: indent(depth), title: path },
        icon("chevron-right", "chevron"),
        icon("folder-fill", "text-warning"),
        el("span", { class: "name", text: label }),
        changedIn(path) > 0 && el("span", { class: "change-badge modified", text: "●", title: `${changedIn(path)} changed` })
      ),
      entries.length
        ? entries.map((e) => (e.type === "dir" ? folder(e.path, baseName(e.path), depth + 1, sorted(children.get(e.path) || [])) : fileRow(e, depth + 1)))
        : el("div", { class: "tree-row text-body-secondary fst-italic", style: indent(depth + 1), text: "empty" })
    );
    details.addEventListener("toggle", () => (details.open ? state.collapsed.delete(path) : state.collapsed.add(path)));
    return details;
  };

  const others = [...state.tree.values()].filter((e) => !rootOf(e.path));
  box.replaceChildren(
    ...TREE_ROOTS.map((root) => folder(root, root, 0, sorted(children.get(root) || []))),
    ...(others.length > 0 ? [folder("(other)", "Other files", 0, sorted(others))] : [])
  );
}

function changedIn(folderPath) {
  let count = 0;
  for (const path of state.changed) if (path.startsWith(folderPath + "/") || (folderPath === "(other)" && !rootOf(path))) count++;
  return count;
}

function fileRow(entry, depth) {
  const kind = changeKind(entry.path);
  return el(
    "button",
    { type: "button", class: `tree-row${state.selected === entry.path ? " active" : ""}`, style: indent(depth), title: entry.path, onclick: () => openFile(entry.path) },
    icon(fileIcon(entry.path)),
    el("span", { class: "name", text: depth === 1 && !rootOf(entry.path) ? entry.path : baseName(entry.path) }),
    kind && el("span", { class: `change-badge ${kind}`, text: kind[0].toUpperCase(), title: kind }),
    entry.size != null && el("span", { class: "size", text: formatBytes(entry.size) })
  );
}

/** "N files changed" with +added -removed lines per file - like `git status`. */
function renderChanges() {
  const box = $("#changes");
  const paths = [...state.changed].sort();
  if (paths.length === 0) return box.replaceChildren();
  box.replaceChildren(
    el("div", { class: "px-2 pt-2 pb-1 fw-semibold small" }, icon("pencil-square", "me-1"), `${paths.length} file${paths.length === 1 ? "" : "s"} changed`),
    ...paths.map((path) => {
      const kind = changeKind(path);
      const stats = diffStats(path);
      return el(
        "button",
        { type: "button", class: `tree-row${state.selected === path ? " active" : ""}`, title: path, onclick: () => openFile(path, { mode: "diff" }) },
        el("span", { class: `change-badge ${kind}`, text: kind[0].toUpperCase(), title: kind }),
        el("span", { class: "name", text: path }),
        stats && el("span", { class: "stat-add", text: `+${stats.added}` }),
        stats && el("span", { class: "stat-del", text: `−${stats.removed}` })
      );
    }),
    el("div", { class: "pb-1" })
  );
}

// ======================================================================
// Viewer
// ======================================================================
function viewerMessage(iconName, ...text) {
  return el("div", { class: "viewer-empty" }, icon(iconName, "fs-2"), el("div", {}, ...text));
}

function renderViewerEmpty() {
  $("#viewer-title").textContent = "No file selected";
  $("#viewer-modes").replaceChildren();
  $("#viewer-download").hidden = true;
  $("#viewer").replaceChildren(
    viewerMessage("file-earmark-code", "Select a file to view it.", el("div", { class: "small mt-1" }, "Files the agent adds or changes are marked ", el("b", { text: "A" }), " / ", el("b", { text: "M" }), " and open as a diff."))
  );
}

async function openFile(path, { mode = null, keepMode = false } = {}) {
  const switched = state.selected !== path;
  state.selected = path;
  if (switched || !keepMode) state.mode = mode;
  renderTree();
  renderChanges();
  $("#viewer-title").textContent = path;
  $("#viewer-title").title = path;

  const exists = state.tree.has(path) || state.touched.has(path);
  if (exists && (!state.contents.has(path) || isStale(path)) && switched) {
    $("#viewer").replaceChildren(viewerMessage("hourglass-split", "Loading…"));
  }
  let content;
  try {
    content = exists ? await ensureContent(path) : { kind: "deleted" };
  } catch (err) {
    if (state.selected === path) $("#viewer").replaceChildren(viewerMessage("exclamation-triangle", errorText(err)));
    return;
  }
  if (state.selected === path) renderViewer(path, content, !switched);
}

function viewerModes(path, content) {
  const modes = [];
  if (diffParts(path)) modes.push("diff");
  if (content.kind === "text") {
    if (extOf(path) === "md") modes.push("preview");
    if (extOf(path) === "csv" || extOf(path) === "tsv") modes.push("table");
    modes.push("code");
  } else if (content.kind !== "deleted" || modes.length === 0) {
    modes.push(content.kind); // image / binary / toolarge / deleted
  }
  return modes;
}

function renderViewer(path, content, keepScroll = false) {
  const modes = viewerModes(path, content);
  if (!modes.includes(state.mode)) state.mode = modes[0]; // a changed file opens as a diff
  const buttons = modes.filter((m) => m in MODE_LABELS);
  $("#viewer-modes").replaceChildren(
    ...(buttons.length < 2 ? [] : buttons)
      .map((m) =>
        el(
          "button",
          { type: "button", class: `btn btn-outline-secondary${m === state.mode ? " active" : ""}`, "aria-pressed": String(m === state.mode), onclick: () => ((state.mode = m), renderViewer(path, content)) },
          MODE_LABELS[m]
        )
      )
  );
  const download = $("#viewer-download");
  download.hidden = !content.bytes;
  download.onclick = () => downloadBytes(baseName(path), content.bytes);

  const box = $("#viewer");
  const scroll = keepScroll ? box.scrollTop : 0;
  box.replaceChildren(renderMode(path, content, state.mode));
  box.scrollTop = scroll;
}

function renderMode(path, content, mode) {
  switch (mode) {
    case "diff":
      return diffView(path);
    case "preview":
      return renderMarkdown(content.text, "markdown-body", dirName(path));
    case "table":
      return csvTable(content.text);
    case "code":
      return codeView(path, content.text);
    case "image":
      return el("div", { class: "viewer-image" }, el("img", { src: content.url, alt: baseName(path) }));
    case "binary":
      return viewerMessage("file-earmark-binary", "This file is not text, so it can't be shown here. Use the download button.");
    case "toolarge":
      return viewerMessage("file-earmark-x", `This file is ${formatBytes(content.size)}: too large to view here (the limit is ${formatBytes(MAX_VIEW_BYTES)}).`);
    default:
      return viewerMessage("trash", "The agent deleted this file.");
  }
}

function codeView(path, text) {
  const lines = text.split("\n");
  if (lines.length > 1 && lines[lines.length - 1] === "") lines.pop();
  const code = el("code", { class: "hljs" });
  const language = LANGUAGES[extOf(path)];
  if (window.hljs && language && hljs.getLanguage(language) && text.length < 300000) {
    // highlight.js escapes the text itself; its output only adds <span class="hljs-..."> tags
    code.innerHTML = hljs.highlight(text, { language, ignoreIllegals: true }).value;
  } else {
    code.textContent = text;
  }
  return el("div", { class: "code" }, el("pre", { class: "gutter", text: lines.map((_, i) => i + 1).join("\n") }), el("pre", {}, code));
}

function diffView(path) {
  const parts = diffParts(path) || [];
  const rows = [];
  parts.forEach((part, index) => {
    const lines = part.value.split("\n");
    if (lines[lines.length - 1] === "") lines.pop();
    if (part.added || part.removed) {
      lines.forEach((line) => rows.push(diffRow(part.added ? "add" : "del", line)));
      return;
    }
    // Unchanged lines: keep a few next to each change, fold the rest
    const head = index === 0 ? 0 : DIFF_CONTEXT;
    const tail = index === parts.length - 1 ? 0 : DIFF_CONTEXT;
    if (lines.length <= head + tail + 1) {
      lines.forEach((line) => rows.push(diffRow("same", line)));
      return;
    }
    lines.slice(0, head).forEach((line) => rows.push(diffRow("same", line)));
    rows.push(el("div", { class: "diff-line diff-gap" }, el("span", { class: "sign" }), `⋯ ${lines.length - head - tail} unchanged lines`));
    lines.slice(lines.length - tail).forEach((line) => rows.push(diffRow("same", line)));
  });
  const note = !state.beforeSizes.has(path)
    ? "New file: every line was added by the agent."
    : !state.tree.has(path)
      ? "Deleted by the agent."
      : "Changes the agent made while answering your last question.";
  return el("div", {}, el("div", { class: "viewer-note", text: note }), el("div", { class: "diff" }, rows));
}

function diffRow(kind, line) {
  const cls = { add: " diff-add", del: " diff-del", same: "" }[kind];
  return el("div", { class: `diff-line${cls}` }, el("span", { class: "sign", text: { add: "+", del: "-", same: " " }[kind] }), el("span", { text: line }));
}

function csvTable(text) {
  const [header = [], ...rows] = parseCsv(text, CSV_PREVIEW_ROWS + 1);
  return el(
    "div",
    {},
    el("div", { class: "viewer-note", text: `First ${rows.length} rows · ${header.length} columns` }),
    el(
      "table",
      { class: "table table-sm table-striped table-hover mb-0 csv-table" },
      el("thead", {}, el("tr", {}, header.map((h) => el("th", { scope: "col", title: h, text: h })))),
      el("tbody", {}, rows.map((row) => el("tr", {}, header.map((_, i) => el("td", { title: row[i] ?? "", text: row[i] ?? "" })))))
    )
  );
}

/** Small CSV parser for previews: quoted fields, "" escapes, CRLF, and , ; tab or | delimiters. */
function parseCsv(text, maxRows) {
  text = text.replace(/^﻿/, "");
  const firstLine = text.slice(0, text.search(/\r?\n|$/));
  const delimiter = [",", ";", "\t", "|"].reduce((best, d) => (firstLine.split(d).length > firstLine.split(best).length ? d : best));

  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length && rows.length < maxRows; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') {
        field += '"';
        i++;
      } else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"' && field === "") quoted = true;
    else if (c === delimiter) {
      row.push(field);
      field = "";
    } else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else field += c;
  }
  if (rows.length < maxRows && (field !== "" || row.length > 0)) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

function downloadBytes(name, bytes) {
  const url = URL.createObjectURL(new Blob([bytes]));
  const link = el("a", { href: url, download: name });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ======================================================================
// Upload + question box
// ======================================================================
function updateCounter() {
  const question = $("#question");
  const counter = $("#char-count");
  counter.textContent = `${question.value.length} / ${question.maxLength}`;
  counter.classList.toggle("text-danger", question.value.length >= question.maxLength);
  // Grow with the text; CSS caps the height, then it scrolls
  question.style.height = "auto";
  question.style.height = `${question.scrollHeight + question.offsetHeight - question.clientHeight}px`;
}

function updateFileHint() {
  const { maxFiles, maxMb } = limits();
  $("#file-hint").textContent = `Up to ${maxFiles} CSV files, ${maxMb} MB each. They are copied to /data in the sandbox.`;
}

function updateControls() {
  const running = state.run !== null;
  const started = state.threadId !== null;
  $("#upload").hidden = started;
  $("#send-btn").hidden = running;
  $("#send-btn").disabled = !state.ready;
  $("#stop-btn").hidden = !running;
  $("#question").readOnly = running;
  $("#refresh-tree").disabled = !started;
  $("#sandbox-label").textContent = started ? `Sandbox ${state.sandboxId}` : "";
  $("#chat-status").replaceChildren(
    running
      ? el("span", {}, el("span", { class: "spinner-grow spinner-grow-sm text-primary me-1", "aria-hidden": "true" }), "Agent is working…")
      : started
        ? `${state.dataFiles.length} data file${state.dataFiles.length === 1 ? "" : "s"}`
        : ""
  );
}

function addFiles(fileList) {
  const { maxFiles, maxMb } = limits();
  for (const file of fileList) {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      showNotice("warning", `${file.name} is not a CSV file.`);
    } else if (file.size > maxMb * 1024 * 1024) {
      showNotice("warning", `${file.name} is larger than ${maxMb} MB.`);
    } else if (state.pending.length >= maxFiles) {
      showNotice("warning", `At most ${maxFiles} files can be uploaded.`);
      break;
    } else if (!state.pending.some((f) => f.name === file.name)) {
      state.pending.push(file);
    }
  }
  renderPendingFiles();
}

function renderPendingFiles() {
  $("#file-list").replaceChildren(
    ...state.pending.map((file, i) =>
      el(
        "li",
        { class: "file-chip" },
        icon("filetype-csv", "text-success"),
        el("span", { class: "text-truncate flex-grow-1", title: file.name, text: file.name }),
        el("span", { class: "size", text: formatBytes(file.size) }),
        el(
          "button",
          { type: "button", class: "btn btn-ghost btn-icon btn-sm", title: "Remove", "aria-label": `Remove ${file.name}`, onclick: () => (state.pending.splice(i, 1), renderPendingFiles()) },
          icon("x-lg")
        )
      )
    )
  );
}

function initDropZone() {
  const zone = $("#drop-zone");
  const input = $("#file-input");
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => {
    addFiles(input.files);
    input.value = ""; // so picking the same file again still fires "change"
  });
  ["dragenter", "dragover"].forEach((type) =>
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((type) => zone.addEventListener(type, () => zone.classList.remove("dragover")));
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    addFiles(event.dataTransfer.files);
  });
  // A file dropped next to the zone would otherwise open in the browser and leave the app
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());
}

// ======================================================================
// Start
// ======================================================================
async function init() {
  applyTheme(document.documentElement.dataset.bsTheme);
  $("#theme-toggle").addEventListener("click", () => {
    const theme = document.documentElement.dataset.bsTheme === "dark" ? "light" : "dark";
    localStorage.setItem(THEME_STORAGE, theme);
    applyTheme(theme);
  });
  initDropZone();
  $("#chat-form").addEventListener("submit", send);
  $("#question").addEventListener("input", updateCounter);
  $("#question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) $("#chat-form").requestSubmit();
  });
  $("#stop-btn").addEventListener("click", stop);
  $("#new-thread").addEventListener("click", newThread);
  $("#refresh-tree").addEventListener("click", () => syncFiles({ tree: true }));

  welcome();
  renderTree();
  renderViewerEmpty();
  updateFileHint();
  updateControls();
  if (await loadStatus()) await restoreThread();
}

init();
