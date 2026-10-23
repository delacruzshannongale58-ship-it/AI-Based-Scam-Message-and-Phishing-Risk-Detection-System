/* ScamScan engine: preprocessing, TF-IDF + Logistic Regression inference, indicator rules, risk scoring.
   Works in the browser and in Node (for testing). The preprocess() function MUST match train_model.py. */
(function (root) {
  "use strict";

  /* ---------- 1. Text preprocessing (mirror of train_model.py) ---------- */
  var URL_RE = /(https?:\/\/\S+|www\.\S+|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:ly|gl|co|xyz|top|click|link|info|site|online|icu|tk|cc|buzz|work|club|ph|com|net|org)(?:\/\S*)?)/g;
  var MONEY_RE = /(?:\u20b1|\$|php\s?)\d[\d,]*(?:\.\d+)?/g;
  var NUM_RE = /\+?\d[\d\s\-]{6,}\d/g;

  function preprocess(text) {
    var t = String(text).toLowerCase();
    t = t.replace(URL_RE, " urltoken ");
    t = t.replace(MONEY_RE, " moneytoken ");
    t = t.replace(NUM_RE, " numtoken ");
    t = t.replace(/[^a-z0-9\s]/g, " ");
    return t.replace(/\s+/g, " ").trim();
  }

  /* ---------- 2. TF-IDF + Logistic Regression inference ---------- */
  var termMap = null, termMapFor = null;
  function predictProbability(text, model) {
    if (!model || !model.terms) return null;
    if (termMapFor !== model) { termMap = new Map(Object.entries(model.terms)); termMapFor = model; }
    var tokens = preprocess(text).match(/\b\w\w+\b/g) || [];
    var counts = new Map();
    function add(term) { counts.set(term, (counts.get(term) || 0) + 1); }
    var useBigrams = model.ngram && model.ngram[1] >= 2;
    for (var i = 0; i < tokens.length; i++) {
      add(tokens[i]);
      if (useBigrams && i + 1 < tokens.length) add(tokens[i] + " " + tokens[i + 1]);
    }
    var feats = [], sumsq = 0;
    counts.forEach(function (c, term) {
      var e = termMap.get(term);
      if (!e) return;
      var w = (1 + Math.log(c)) * e[0];          // sublinear TF x IDF
      feats.push([w, e[1]]); sumsq += w * w;
    });
    var norm = Math.sqrt(sumsq) || 1, z = model.intercept;
    for (var j = 0; j < feats.length; j++) z += (feats[j][0] / norm) * feats[j][1];
    return 1 / (1 + Math.exp(-z));
  }

  /* ---------- 3. Rule-based scam indicators ---------- */
  var BRANDS = "gcash|maya|paymaya|bdo|bpi|metrobank|landbank|unionbank|security bank|pnb|rcbc|lbc|j&t|jnt|flash express|ninja van|shopee|lazada|grab|globe|smart|dito|bir|sss|philhealth|pag-?ibig|lto|nbi|pnp|dict|paypal|facebook|meta|netflix|amazon|dhl|fedex|apple id|apple|google|microsoft|bureau of internal revenue|customer (?:service|support)|tech(?:nical)? support";
  var SECRET = "otp|one[- ]time (?:password|pin|code)|pin(?: code)?|password|passcode|cvv|verification code|security code|mpin";

  var INDICATORS = [
    { id: "urgency", name: "Urgency / pressure", points: 2,
      explain: "Scammers rush you so you act before thinking. Real organizations rarely demand action within hours.",
      patterns: [/\b(?:within\s+(?:the\s+next\s+)?\d+\s*(?:hours?|hrs?|minutes?|mins?|days?)|24\s*hours?|immediately|right away|act now|urgent(?:ly)?|last (?:warning|chance|notice)|final (?:warning|notice)|expires? (?:today|soon|tonight)|before (?:they )?expire|today only|limited offer|asap|as soon as possible|agad|ngayon na|sa loob ng \d+ oras)\b/gi] },
    { id: "link", name: "Suspicious link", points: 3,
      explain: "Shortened, odd-looking, or look-alike web addresses are commonly used to hide fake login pages.",
      patterns: [
        /\b(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|cutt\.ly|rb\.gy|shorturl\.at|ow\.ly|s\.id|tiny\.cc)\/?\S*/gi,
        /https?:\/\/\d{1,3}(?:\.\d{1,3}){3}\S*/gi,
        /\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:xyz|top|click|link|info|site|online|icu|tk|cc|buzz|work|club)\b(?:\/\S*)?/gi,
        new RegExp("\\b[a-z0-9.-]*(?:gcash|maya|paymaya|bdo|bpi|metrobank|landbank|unionbank|lbc|jnt|shopee|lazada|globe|smart|paypal|facebook|netflix|apple|google)[a-z0-9.-]*[-.](?:verify|secure|login|update|claim|support|account|confirm|redeliver\\w*|delivery|gift|voucher)[a-z0-9.-]*\\.[a-z]{2,}\\S*", "gi")] },
    { id: "otp", name: "Asks for OTP / password", points: 4, negatable: true,
      explain: "No bank, e-wallet, or agency will ask you to send your OTP, PIN, or password. Anyone who asks is likely a scammer.",
      patterns: [
        new RegExp("\\b(?:send|share|give|provide|reply(?: with)?|tell|confirm|ibigay|ipadala|i-send)\\b[^.!?\\n]{0,40}?\\b(?:" + SECRET + ")\\b", "gi"),
        new RegExp("\\b(?:" + SECRET + ")\\b[^.!?\\n]{0,30}?\\b(?:send|share|reply|provide|ibigay|ipadala)\\b", "gi")] },
    { id: "financial", name: "Asks for personal / financial information", points: 3,
      explain: "Requests for card numbers, bank details, ID photos, or birth details are used for identity theft and account takeover.",
      patterns: [
        /\b(?:card number|credit card|debit card|account (?:number|details)|bank account(?: (?:number|details))?|atm (?:pin|card)|birth ?date|date of birth|mother'?s maiden name|id (?:number|picture|photo)|selfie with id|sss number|full name and address|personal (?:details|information)|mobile number and otp)\b/gi,
        /\b(?:send|enter|provide|reply|submit|confirm|update|verify|ibigay)\b[^.!?\n]{0,50}?\b(?:card|account|bank|personal|id|birth|address)\b[^.!?\n]{0,20}?\b(?:number|details|info(?:rmation)?|date|photo|picture)\b/gi] },
    { id: "prize", name: "Prize or reward bait", points: 2,
      explain: "Surprise prizes, free gifts, and rewards are bait to get you to click or share information.",
      patterns: [/\b(?:you (?:have )?(?:won|win)|you(?:'re| are) (?:a |the )?(?:lucky )?winner|lucky (?:winner|draw)|congratulations|congrats|grand prize|free (?:gift|load|money|cash|data|iphone|netflix|entry)|claim (?:your |now|the )?(?:prize|reward|gift|cash|bonus|voucher|load)?|raffle|prize draw|reward points|cash bonus|unclaimed prize|nanalo ka|premyo)\b/gi] },
    { id: "threat", name: "Threat or account suspension", points: 2,
      explain: "Threats of blocked accounts, legal action, or arrest create fear so you stop thinking carefully.",
      patterns: [/\b(?:account (?:will be |has been |is |was )?(?:suspended|blocked|locked|closed|deactivated|limited|disabled|deleted)|(?:card|wallet|sim) (?:will be |has been |is )?(?:blocked|deactivated|suspended)|(?:suspend|block|lock|close|deactivate)(?:ed)? your account|legal action|warrant of arrest|face a warrant|arrest|will be (?:terminated|deleted|closed|disabled|deactivated)|failure to (?:comply|verify|respond)|mai-block|mablock)\b/gi] },
    { id: "impersonation", name: "Pretends to be a trusted organization", points: 2, conditional: true,
      explain: "Scammers copy the names of real banks, e-wallets, couriers, and agencies. Always verify through the official app or hotline.",
      patterns: [new RegExp("\\b(?:" + BRANDS + ")\\b", "gi")] },
    { id: "payment", name: "Asks for a fee or payment", points: 3,
      explain: "Being asked to pay a fee to release a parcel, loan, or prize is a classic advance-fee scam.",
      patterns: [
        /\b(?:release|processing|customs|clearance|redelivery|shipping|handling|insurance|delivery) fee\b/gi,
        /\b(?:pay|send|transfer|deposit|remit)\b[^.!?\n]{0,40}?\b(?:fee|fees|tax|payment|charge|load|money|pesos?)\b/gi,
        /\bsend (?:it )?(?:back )?to this (?:gcash |maya |bank )?(?:number|account)\b/gi] },
    { id: "greeting", name: "Generic greeting", points: 1,
      explain: "Mass scam messages often use impersonal openers because the sender does not know your name.",
      patterns: [/\b(?:dear (?:customer|user|client|member|valued\s+\w+|account holder|sir\/?\s?madam|beneficiary)|valued customer|hi sir\/madam|hello sir\/madam)\b/gi] }
  ];

  function findAll(re, text) {
    var r = new RegExp(re.source, re.flags.indexOf("g") >= 0 ? re.flags : re.flags + "g"), out = [], m;
    while ((m = r.exec(text)) !== null) {
      out.push({ start: m.index, end: m.index + m[0].length, text: m[0] });
      if (m[0].length === 0) r.lastIndex++;
    }
    return out;
  }
  function negated(text, start) {
    var before = text.slice(Math.max(0, start - 25), start);
    return /(?:do not|don'?t|dont|never|huwag|wag)\s+(?:\w+\s+){0,2}$/i.test(before);
  }

  function detectIndicators(text) {
    var found = [];
    INDICATORS.forEach(function (ind) {
      var ms = [];
      ind.patterns.forEach(function (re) {
        findAll(re, text).forEach(function (m) {
          if (ind.negatable && negated(text, m.start)) return;
          ms.push(m);
        });
      });
      if (ms.length) found.push({ id: ind.id, name: ind.name, points: ind.points, explain: ind.explain, conditional: !!ind.conditional, matches: ms });
    });
    // brand names alone are not suspicious: keep only if another warning sign exists
    var others = found.filter(function (f) { return !f.conditional && f.id !== "greeting"; }).length;
    return found.filter(function (f) { return !f.conditional || others > 0; });
  }

  /* ---------- 4. Risk scoring ---------- */
  var RECOMMENDATIONS = {
    High: ["Do not click any link or reply to this message.", "Never share an OTP, password, or PIN with anyone.", "Contact the organization using its official app or hotline, not the details in the message.", "Consider reporting the message to the platform or the authorities."],
    Medium: ["Be careful and do not click links yet.", "Verify the sender through official channels before taking any action.", "Never share OTPs, passwords, or card details."],
    Low: ["No strong warning signs were found, but this does not prove the message is safe.", "Stay cautious with unexpected links, payment requests, or requests for codes."]
  };

  function analyze(text, model) {
    var clean = String(text || "").trim();
    if (clean.length < 3) return null;
    var p = predictProbability(clean, model);
    var inds = detectIndicators(clean);
    var points = inds.reduce(function (a, i) { return a + i.points; }, 0);
    var R = Math.min(points / 10, 1);
    var score = p === null ? R : 0.6 * p + 0.4 * R;
    var hasOtp = inds.some(function (i) { return i.id === "otp"; });
    if (hasOtp && score < 0.35) score = 0.35;               // safety override: OTP/password requests are never Low
    var level = score < 0.35 ? "Low" : score < 0.65 ? "Medium" : "High";
    return { text: clean, p: p, points: points, R: R, score: score, level: level,
             label: level === "Low" ? "Potentially Safe" : "Suspicious",
             indicators: inds, advice: RECOMMENDATIONS[level], modelUsed: p !== null };
  }

  /* ---------- 5. Highlight ranges ---------- */
  function highlightRanges(result) {
    var all = [];
    result.indicators.forEach(function (ind) { ind.matches.forEach(function (m) { all.push({ start: m.start, end: m.end, id: ind.id }); }); });
    all.sort(function (a, b) { return a.start - b.start || b.end - a.end; });
    var out = [], last = -1;
    all.forEach(function (r) { if (r.start >= last) { out.push(r); last = r.end; } });
    return out;
  }

  var api = { preprocess: preprocess, predictProbability: predictProbability, detectIndicators: detectIndicators,
              analyze: analyze, highlightRanges: highlightRanges, INDICATORS: INDICATORS };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.ScamEngine = api;
})(typeof window !== "undefined" ? window : globalThis);
