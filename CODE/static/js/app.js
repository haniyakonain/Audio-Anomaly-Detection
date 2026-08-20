// FanSound AI: shared front-end behaviour (tabs, dropzone, sample library)

document.addEventListener("DOMContentLoaded", function () {
  initTabs();
  initDropzone();
  initSampleLibrary();
});

/* ---------- Tabs ---------- */
function initTabs() {
  const buttons = document.querySelectorAll("[data-fs-tab]");
  if (!buttons.length) return;

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-fs-tab");

      buttons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      document.querySelectorAll(".fs-tab-panel").forEach((panel) => {
        panel.classList.toggle("active", panel.id === target);
      });
    });
  });
}

/* ---------- Upload dropzone ---------- */
function initDropzone() {
  const dropzone = document.getElementById("fs-dropzone");
  const input = document.getElementById("fs-file-input");
  const chip = document.getElementById("fs-file-chip");
  const chipName = document.getElementById("fs-file-chip-name");
  const clearBtn = document.getElementById("fs-file-clear");
  const submitBtn = document.getElementById("fs-upload-submit");
  if (!dropzone || !input) return;

  function setFile(file) {
    if (!file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    if (chipName) chipName.textContent = file.name;
    if (chip) chip.classList.add("show");
    if (submitBtn) submitBtn.disabled = false;
  }

  dropzone.addEventListener("click", () => input.click());

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });

  input.addEventListener("change", () => {
    if (input.files && input.files[0]) setFile(input.files[0]);
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) setFile(file);
  });

  if (clearBtn) {
    clearBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      input.value = "";
      if (chip) chip.classList.remove("show");
      if (submitBtn) submitBtn.disabled = true;
    });
  }
}

/* ---------- Sample library: filtering + "predict this" ---------- */
function initSampleLibrary() {
  const grid = document.getElementById("fs-sample-grid");
  if (!grid) return;

  const chips = document.querySelectorAll("[data-fs-filter]");
  const cards = grid.querySelectorAll(".fs-sample-card");

  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      chips.forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      const filter = chip.getAttribute("data-fs-filter");

      cards.forEach((card) => {
        const matches =
          filter === "all" ||
          card.getAttribute("data-machine") === filter ||
          card.getAttribute("data-label") === filter;
        card.style.display = matches ? "" : "none";
      });
    });
  });

  grid.querySelectorAll("[data-fs-predict-form]").forEach((form) => {
    form.addEventListener("submit", () => {
      const btn = form.querySelector("button[type=submit]");
      if (btn) {
        btn.disabled = true;
        btn.textContent = "Predicting…";
      }
    });
  });
}
