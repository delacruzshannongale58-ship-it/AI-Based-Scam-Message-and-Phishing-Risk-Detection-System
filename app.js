/* ScamScan UI: paste/upload/text input -> OCR -> engine -> result screen. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var el = { drop: $("dropZone"), preview: $("preview"), pasteBtn: $("pasteBtn"), uploadBtn: $("uploadBtn"), clearBtn: $("clearBtn"),
    file: $("fileInput"), text: $("msgText"), status: $("status"), bar: $("bar"), fill: $("barFill"), result: $("result"),
    badge: $("badge"), risk: $("risk"), meter: $("meterFill"), score: $("scoreLine"), hl: $("highlighted"),
    signs: $("signs"), advice: $("advice"), how: $("howText"), chip: $("modelChip") };

  var MODEL = (typeof SCAM_MODEL !== "undefined") ? SCAM_MODEL : null;
  el.chip.textContent = MODEL ? "AI model loaded" : "rules only";
  var worker = null, workerPromise = null, jobId = 0, debounceTimer = null, lastURL = null;

  function esc(s) { return s.replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }
  function setStatus(msg, pct) {
    el.status.textContent = msg;
    if (typeof pct === "number") { el.bar.hidden = false; el.fill.style.width = Math.round(pct * 100) + "%"; } else { el.bar.hidden = true; }
  }

  /* ---------- rendering ---------- */
  function render(r) {
    if (!r) { el.result.hidden = true; return; }
    el.result.hidden = false;
    el.result.className = "card result " + r.level.toLowerCase();
    el.badge.textContent = r.label.toUpperCase();
    el.risk.textContent = r.level.toUpperCase() + " RISK";
    el.meter.style.width = Math.max(4, Math.round(r.score * 100)) + "%";
    el.score.textContent = "Risk score: " + r.score.toFixed(2) + "  (AI model " + (r.p === null ? "n/a" : r.p.toFixed(2)) + ", warning-sign points " + r.points + ")";

    var ranges = ScamEngine.highlightRanges(r), html = "", pos = 0;
    ranges.forEach(function (g) {
      html += esc(r.text.slice(pos, g.start)) + '<mark class="m-' + g.id + '">' + esc(r.text.slice(g.start, g.end)) + "</mark>";
      pos = g.end;
    });
    el.hl.innerHTML = html + esc(r.text.slice(pos));

    el.signs.innerHTML = "";
    if (!r.indicators.length) {
      el.signs.innerHTML = '<li class="none">No warning signs were detected. That does not prove the message is safe.</li>';
    } else {
      r.indicators.forEach(function (i) {
        var li = document.createElement("li");
        var ex = i.matches.slice(0, 3).map(function (m) { return "\u201C" + m.text.trim() + "\u201D"; }).join(", ");
        li.innerHTML = "<b>" + esc(i.name) + "</b><div class='ex'>Found: " + esc(ex) + "</div><div class='why'>" + esc(i.explain) + "</div>";
        el.signs.appendChild(li);
      });
    }
    el.advice.innerHTML = r.advice.map(function (a) { return "<li>" + esc(a) + "</li>"; }).join("");
    el.how.textContent = "The AI model (TF-IDF + Logistic Regression) estimated " + (r.p === null ? "nothing (model file missing)" : "a " + Math.round(r.p * 100) + "% chance") +
      " that this message is suspicious. The rule engine found " + r.indicators.length + " warning sign(s) worth " + r.points +
      " point(s). Final score = 0.6 \u00D7 model + 0.4 \u00D7 min(points/10, 1) = " + r.score.toFixed(2) +
      ". Below 0.35 is Low, 0.35\u20130.65 is Medium, above 0.65 is High. Any request for an OTP or password is never rated Low.";
  }

  function analyzeText(text) {
    var r = ScamEngine.analyze(text, MODEL);
    if (!r) { render(null); setStatus("Waiting for a message…"); return; }
    render(r); setStatus("Analysis complete.");
  }

  /* ---------- OCR ---------- */
  function getWorker() {
    if (worker) return Promise.resolve(worker);
    if (!window.Tesseract) return Promise.reject(new Error("OCR library did not load. Check your internet connection."));
    if (!workerPromise) {
      workerPromise = Tesseract.createWorker("eng", 1, { logger: function (m) {
        if (m.status === "recognizing text") setStatus("Reading message… " + Math.round(m.progress * 100) + "%", m.progress);
        else if (m.status) setStatus("Preparing reader… (" + m.status + ")", typeof m.progress === "number" ? m.progress : undefined);
      } }).then(function (w) { worker = w; return w; });
    }
    return workerPromise;
  }

  function loadImage(url) {
    return new Promise(function (res, rej) { var im = new Image(); im.onload = function () { res(im); }; im.onerror = function () { rej(new Error("Could not read this image.")); }; im.src = url; });
  }
  // grayscale, upscale small images, and invert dark-mode screenshots so OCR reads them better
  function prepare(im) {
    var scale = im.width < 1000 ? 1000 / im.width : 1; if (im.width * scale > 2200) scale = 2200 / im.width;
    var c = document.createElement("canvas"); c.width = Math.round(im.width * scale); c.height = Math.round(im.height * scale);
    var ctx = c.getContext("2d"); ctx.drawImage(im, 0, 0, c.width, c.height);
    var d = ctx.getImageData(0, 0, c.width, c.height), px = d.data, sum = 0, n = px.length / 4, i;
    for (i = 0; i < px.length; i += 4) { var g = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]; px[i] = px[i + 1] = px[i + 2] = g; sum += g; }
    if (sum / n < 110) for (i = 0; i < px.length; i += 4) { px[i] = px[i + 1] = px[i + 2] = 255 - px[i]; }
    ctx.putImageData(d, 0, 0); return c;
  }

  function handleImage(blob) {
    if (!blob || !/^image\//.test(blob.type || "image/png")) { setStatus("That file is not an image. Please paste or upload a JPG or PNG screenshot."); return; }
    if (blob.size > 15 * 1024 * 1024) { setStatus("Image is too large (max 15 MB)."); return; }
    var id = ++jobId;
    if (lastURL) URL.revokeObjectURL(lastURL);
    lastURL = URL.createObjectURL(blob);
    el.preview.src = lastURL; el.preview.hidden = false;
    setStatus("Screenshot received. Reading message…", 0.02); render(null);
    loadImage(lastURL).then(prepare).then(function (canvas) {
      return getWorker().then(function (w) { return w.recognize(canvas); });
    }).then(function (out) {
      if (id !== jobId) return;                                   // a newer paste replaced this one
      var text = (out.data.text || "").replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
      if (text.length < 3) { el.text.value = ""; setStatus("No readable text found. Try a clearer or larger screenshot, or type the message."); return; }
      el.text.value = text; analyzeText(text);
    }).catch(function (e) { if (id === jobId) setStatus("Error: " + e.message); });
  }

  /* ---------- inputs ---------- */
  function imageFromItems(items) {
    for (var i = 0; i < items.length; i++) if (items[i].kind === "file" && /^image\//.test(items[i].type)) return items[i].getAsFile();
    return null;
  }
  document.addEventListener("paste", function (e) {
    var cd = e.clipboardData; if (!cd) return;
    var img = imageFromItems(cd.items || []);
    if (img) { e.preventDefault(); handleImage(img); return; }
    // plain text pasted anywhere except the text box: send it to the text box
    if (document.activeElement !== el.text) {
      var t = cd.getData("text/plain");
      if (t) { e.preventDefault(); el.text.value = t; el.preview.hidden = true; analyzeText(t); }
    }
  });

  el.pasteBtn.addEventListener("click", function () {
    if (!navigator.clipboard || !navigator.clipboard.read) { setStatus("Your browser cannot read the clipboard from a button. Long-press the text box and choose Paste, or use Upload image."); return; }
    navigator.clipboard.read().then(function (items) {
      for (var i = 0; i < items.length; i++) {
        var types = items[i].types;
        for (var j = 0; j < types.length; j++) if (/^image\//.test(types[j])) { return items[i].getType(types[j]).then(handleImage); }
      }
      return navigator.clipboard.readText().then(function (t) {
        if (!t) { setStatus("The clipboard has no image or text. Copy a screenshot first."); return; }
        el.text.value = t; el.preview.hidden = true; analyzeText(t);
      });
    }).catch(function () { setStatus("Clipboard access was blocked. Long-press the text box and choose Paste, or use Upload image."); });
  });
  el.uploadBtn.addEventListener("click", function () { el.file.click(); });
  el.file.addEventListener("change", function () { if (el.file.files[0]) handleImage(el.file.files[0]); el.file.value = ""; });
  el.drop.addEventListener("click", function () { el.drop.focus(); });
  ["dragenter", "dragover"].forEach(function (ev) { el.drop.addEventListener(ev, function (e) { e.preventDefault(); el.drop.classList.add("over"); }); });
  ["dragleave", "drop"].forEach(function (ev) { el.drop.addEventListener(ev, function (e) { e.preventDefault(); el.drop.classList.remove("over"); }); });
  el.drop.addEventListener("drop", function (e) { var f = e.dataTransfer && e.dataTransfer.files[0]; if (f) handleImage(f); });

  el.text.addEventListener("input", function () {                  // live analysis while typing or editing
    clearTimeout(debounceTimer); jobId++;
    setStatus("Analyzing…");
    debounceTimer = setTimeout(function () { analyzeText(el.text.value); }, 500);
  });
  el.clearBtn.addEventListener("click", function () {
    jobId++; clearTimeout(debounceTimer); el.text.value = ""; el.preview.hidden = true; render(null); setStatus("Waiting for a message…");
  });

  window.addEventListener("load", function () { if (window.Tesseract) getWorker().then(function () { setStatus("Ready. Paste a screenshot or text."); }).catch(function () { setStatus("Ready for text. The screenshot reader needs an internet connection."); }); });
})();
