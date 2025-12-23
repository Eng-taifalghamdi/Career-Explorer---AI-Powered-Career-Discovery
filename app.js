import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const runBtn = document.getElementById("runBtn");
const form = document.getElementById("qaForm");
const resultsCard = document.getElementById("resultsCard");
const progressContainer = document.getElementById("progressContainer");
const progressFill = document.getElementById("progressFill");
const progressText = document.getElementById("progressText");
const modal = document.getElementById("explanationModal");
const modalTitle = document.getElementById("modalTitle");
const modalBody = document.getElementById("modalBody");
const modalClose = document.getElementById("modalClose");
const exportBtn = document.getElementById("exportBtn");

// ======================
// Configuration
// ======================
const SIM_THRESHOLD_STRICT = {
  skills: 0.42,
  knowledge: 0.40,
  tasks: 0.40,
  occ: 0.38,
};

const SIM_THRESHOLD_RELAXED = {
  skills: 0.38,
  knowledge: 0.36,
  tasks: 0.36,
  occ: 0.34,
};

const SIM_THRESHOLD_MIN = {
  skills: 0.35,
  knowledge: 0.33,
  tasks: 0.33,
  occ: 0.31,
};

const WEIGHTS = {
  skills: 0.35,
  tasks: 0.35,
  knowledge: 0.20,
  occ: 0.10,
};

const PREF_BOOST_MATCH = 0.10;
const PREF_BOOST_MISMATCH = -0.15;

const MIN_MATCHES_STRICT = 2;
const MIN_MATCHES_NORMAL = 1;
const MIN_MATCHES_RELAXED = 1;

const FINAL_TOP_N = 10;

// ======================
// Globals
// ======================
let embedder = null;
const EMB_DIR = "./data/embeddings";
let store = null;
let currentUserAnswers = null; // Store for explanations

// ======================
// Progress Management
// ======================
function showProgress(text, percent = null) {
  progressContainer.style.display = "block";
  progressText.textContent = text;
  if (percent !== null) {
    progressFill.style.width = `${percent}%`;
  }
}

function hideProgress() {
  progressContainer.style.display = "none";
}

function setStatus(msg) {
  statusEl.textContent = msg;
}

// ======================
// Core Functions (same as before)
// ======================
function dotCosineNormalized(userVec, data, offset, cols) {
  let dot = 0;
  for (let c = 0; c < cols; c++) dot += userVec[c] * data[offset + c];
  return dot;
}

async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  return res.json();
}

async function loadNPY(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  const buf = await res.arrayBuffer();
  const dv = new DataView(buf);

  const major = dv.getUint8(6);
  const headerLen = major <= 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const headerStart = major <= 1 ? 10 : 12;

  const headerBytes = new Uint8Array(buf, headerStart, headerLen);
  const headerText = new TextDecoder().decode(headerBytes);

  const shapeMatch = headerText.match(/\(\s*(\d+)\s*,\s*(\d+)\s*\)/);
  if (!shapeMatch) throw new Error("Could not parse .npy shape");

  const rows = parseInt(shapeMatch[1], 10);
  const cols = parseInt(shapeMatch[2], 10);

  const dataStart = headerStart + headerLen;
  const floats = new Float32Array(buf, dataStart, rows * cols);

  return { rows, cols, data: floats };
}

function toVector(out) {
  if (out?.data && out?.dims) {
    const { data, dims } = out;
    if (dims.length === 1) return Array.from(data);

    const [tokens, hidden] = dims;
    const vec = new Array(hidden).fill(0);
    for (let t = 0; t < tokens; t++) {
      for (let h = 0; h < hidden; h++) vec[h] += data[t * hidden + h];
    }
    for (let h = 0; h < hidden; h++) vec[h] /= tokens;
    return vec;
  }

  if (Array.isArray(out)) {
    if (Array.isArray(out[0])) return out[0];
    return out;
  }

  return Array.from(out);
}

async function embedText(text) {
  const out = await embedder(text, { pooling: "mean", normalize: true });
  return toVector(out);
}

function domainLabel(domain) {
  switch (domain) {
    case "skills": return "Skills";
    case "knowledge": return "Knowledge";
    case "tasks": return "Tasks";
    case "occ": return "Work Style";
    default: return domain;
  }
}

function extractKeyTerms(text) {
  const commonWords = new Set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
    'i', 'me', 'my', 'we', 'you', 'it', 'that', 'this', 'what', 'which',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'might', 'can', 'like', 'also', 'very', 'really', 'want', 'need'
  ]);

  const words = text.toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 2 && !commonWords.has(w));

  return [...new Set(words)];
}

function calculateDiversity(results) {
  const titleWords = new Set();
  
  return results.map(r => {
    const words = r.title.toLowerCase().split(/\s+/);
    let novelty = 0;
    
    words.forEach(word => {
      if (word.length > 3 && !titleWords.has(word)) {
        novelty++;
        titleWords.add(word);
      }
    });
    
    return { ...r, diversityScore: novelty };
  });
}

function getAppliedBoost(title, userAnswers) {
  const t = title.toLowerCase();
  const allText = Object.values(userAnswers).join(' ').toLowerCase();
  
  const mentionsPractical = 
    allText.includes('build') || allText.includes('creat') || 
    allText.includes('cod') || allText.includes('develop') ||
    allText.includes('mak') || allText.includes('design') ||
    allText.includes('write') || allText.includes('program') ||
    allText.includes('app') || allText.includes('website') ||
    allText.includes('software') || allText.includes('project');
  
  const onlyTheoretical = 
    !mentionsPractical && (
      allText.includes('math') || allText.includes('statistic') ||
      allText.includes('theory') || allText.includes('research')
    );
  
  const isPureTheory = 
    t.includes('mathematician') || t.includes('statistician') ||
    (t.includes('research') && t.includes('scientist')) ||
    t.includes('theoretical');
  
  const isAppliedTech = 
    t.includes('software') || t.includes('developer') || 
    t.includes('engineer') || t.includes('programmer') ||
    t.includes('web ') || t.includes('application');
  
  let boost = 0;
  
  if (mentionsPractical && isAppliedTech) boost += 0.15;
  if (mentionsPractical && isPureTheory) boost -= 0.20;
  if (onlyTheoretical && isPureTheory) boost += 0.10;
  
  return boost;
}

function readPreferences() {
  const office = document.getElementById("prefOffice")?.checked ?? false;
  const field = document.getElementById("prefField")?.checked ?? false;
  const industrial = document.getElementById("prefIndustrial")?.checked ?? false;
  const military = document.getElementById("prefMilitary")?.checked ?? false;
  const healthcare = document.getElementById("prefHealthcare")?.checked ?? false;
  const tech = document.getElementById("prefTech")?.checked ?? false;
  const creative = document.getElementById("prefCreative")?.checked ?? false;
  const education = document.getElementById("prefEducation")?.checked ?? false;
  const business = document.getElementById("prefBusiness")?.checked ?? false;

  const any = office || field || industrial || military || healthcare || tech || creative || education || business;
  return { office, field, industrial, military, healthcare, tech, creative, education, business, any };
}

function classifyTitle(title) {
  const t = (title || "").toLowerCase();

  // Military
  const isMilitary =
    t.includes("military") || t.includes("infantry") || t.includes("enlisted") ||
    t.includes("tactical") || t.includes("command") || t.includes("special forces") ||
    t.includes("army") || t.includes("navy") || t.includes("air force");

  // Healthcare
  const isHealthcare =
    t.includes("physician") || t.includes("doctor") || t.includes("nurse") ||
    t.includes("medical") || t.includes("health") || t.includes("therapist") ||
    t.includes("dental") || t.includes("surgeon") || t.includes("practitioner") ||
    t.includes("pharmacist") || t.includes("paramedic") || t.includes("radiologic") ||
    t.includes("diagnostic") || t.includes("clinical") || t.includes("patient care");

  // Tech/IT
  const isTech =
    t.includes("software") || t.includes("developer") || t.includes("programmer") ||
    t.includes("engineer") && (t.includes("software") || t.includes("computer") || t.includes("network")) ||
    t.includes("data scientist") || t.includes("data analyst") || t.includes("web ") ||
    t.includes("database") || t.includes("systems analyst") || t.includes("information technology") ||
    t.includes("cybersecurity") || t.includes("it ") || t.includes("tech ");

  // Creative/Arts
  const isCreative =
    t.includes("designer") || t.includes("artist") || t.includes("photographer") ||
    t.includes("writer") || t.includes("editor") || t.includes("graphic") ||
    t.includes("media") || t.includes("creative") || t.includes("animator") ||
    t.includes("illustrator") || t.includes("musician") || t.includes("producer") ||
    t.includes("director") && !t.includes("executive") || t.includes("actor");

  // Education/Teaching
  const isEducation =
    t.includes("teacher") || t.includes("professor") || t.includes("instructor") ||
    t.includes("educator") || t.includes("tutor") || t.includes("lecturer") ||
    t.includes("trainer") || t.includes("coach") && !t.includes("athletic") ||
    t.includes("librarian") || t.includes("counselor") && t.includes("school");

  // Business/Finance
  const isBusiness =
    t.includes("accountant") || t.includes("financial") || t.includes("finance") ||
    t.includes("accounting") || t.includes("auditor") || t.includes("banker") ||
    t.includes("consultant") || t.includes("sales") || t.includes("marketing") ||
    t.includes("business analyst") || t.includes("economist") || t.includes("actuary") ||
    t.includes("investment") || t.includes("portfolio") || t.includes("broker") ||
    t.includes("underwriter") || t.includes("tax") && !t.includes("taxi");

  // Industrial
  const isIndustrial =
    t.includes("assembler") || t.includes("welder") || t.includes("fabricator") ||
    t.includes("machinist") || t.includes("operator") && !t.includes("computer") ||
    t.includes("pumper") || t.includes("loader") || t.includes("truck") || 
    t.includes("driver") || t.includes("drilling") || t.includes("mining") || 
    t.includes("refuse") || t.includes("production") || t.includes("manufacturing") || 
    t.includes("warehouse") || t.includes("forklift") || t.includes("mechanic") || 
    t.includes("maintenance") && !t.includes("software") ||
    t.includes("janitorial") || t.includes("housekeeping") || t.includes("custodian") ||
    t.includes("cook") || t.includes("food service") || t.includes("cafeteria") ||
    t.includes("laborer") || t.includes("construction");

  // Field
  const isField =
    (t.includes("field") || t.includes("technician") && !isHealthcare && !isTech ||
    t.includes("installation") || t.includes("service") && !t.includes("food") ||
    t.includes("repair") || t.includes("inspector")) && !isIndustrial;

  // Office (default for professional roles)
  const isOffice = !isMilitary && !isHealthcare && !isTech && !isCreative && 
                   !isEducation && !isBusiness && !isIndustrial && !isField;

  return { isOffice, isField, isIndustrial, isMilitary, isHealthcare, isTech, isCreative, isEducation, isBusiness };
}

function preferenceBoost(title, prefs) {
  if (!prefs.any) return 0;

  const cls = classifyTitle(title);
  let boost = 0;

  // Apply boosts for matches
  if (prefs.office && cls.isOffice) boost += PREF_BOOST_MATCH;
  if (prefs.field && cls.isField) boost += PREF_BOOST_MATCH;
  if (prefs.industrial && cls.isIndustrial) boost += PREF_BOOST_MATCH;
  if (prefs.military && cls.isMilitary) boost += PREF_BOOST_MATCH;
  if (prefs.healthcare && cls.isHealthcare) boost += PREF_BOOST_MATCH;
  if (prefs.tech && cls.isTech) boost += PREF_BOOST_MATCH;
  if (prefs.creative && cls.isCreative) boost += PREF_BOOST_MATCH;
  if (prefs.education && cls.isEducation) boost += PREF_BOOST_MATCH;
  if (prefs.business && cls.isBusiness) boost += PREF_BOOST_MATCH;

  // Apply penalties for mismatches
  if (!prefs.office && cls.isOffice && prefs.any) boost += PREF_BOOST_MISMATCH;
  if (!prefs.field && cls.isField && prefs.any) boost += PREF_BOOST_MISMATCH;
  if (!prefs.industrial && cls.isIndustrial && prefs.any) boost += PREF_BOOST_MISMATCH;
  if (!prefs.military && cls.isMilitary && prefs.any) boost += PREF_BOOST_MISMATCH;
  if (!prefs.healthcare && cls.isHealthcare && prefs.any) boost += PREF_BOOST_MISMATCH;
  if (!prefs.tech && cls.isTech && prefs.any) boost += PREF_BOOST_MISMATCH;
  if (!prefs.creative && cls.isCreative && prefs.any) boost += PREF_BOOST_MISMATCH;
  if (!prefs.education && cls.isEducation && prefs.any) boost += PREF_BOOST_MISMATCH;
  if (!prefs.business && cls.isBusiness && prefs.any) boost += PREF_BOOST_MISMATCH;

  return boost;
}

function topKBySimilarityDomain(userVec, domain, k, thresholdSet) {
  const { meta, npy } = store[domain];
  const { rows, cols, data } = npy;

  if (userVec.length !== cols) {
    throw new Error(`Embedding dim mismatch for ${domain}`);
  }

  const threshold = thresholdSet[domain] ?? 0.0;
  const candidates = [];

  for (let r = 0; r < rows; r++) {
    const offset = r * cols;
    const score = dotCosineNormalized(userVec, data, offset, cols);

    if (score < threshold) continue;

    candidates.push({
      id: meta[r].id,
      title: meta[r].title,
      score,
    });
  }

  candidates.sort((a, b) => b.score - a.score);
  return candidates.slice(0, k);
}

function performSearch(userVecs, thresholdSet, minMatches, topK) {
  const topLists = {
    skills: topKBySimilarityDomain(userVecs.skills, "skills", topK, thresholdSet),
    knowledge: topKBySimilarityDomain(userVecs.knowledge, "knowledge", topK, thresholdSet),
    tasks: topKBySimilarityDomain(userVecs.tasks, "tasks", topK, thresholdSet),
    occ: topKBySimilarityDomain(userVecs.occ, "occ", topK, thresholdSet),
  };

  const map = new Map();

  for (const [domain, list] of Object.entries(topLists)) {
    const w = WEIGHTS[domain] ?? 0.25;

    for (const it of list) {
      const key = it.id;
      if (!map.has(key)) {
        map.set(key, {
          id: key,
          title: it.title,
          matchCount: 0,
          domains: [],
          scores: [],
          weightedScore: 0,
        });
      }
      const entry = map.get(key);
      entry.matchCount += 1;
      entry.domains.push(domain);
      entry.scores.push(it.score);
      entry.weightedScore += w * it.score;
    }
  }

  return Array.from(map.values()).filter(r => r.matchCount >= minMatches);
}

// ======================
// Enhanced LLM-style Explanation Generator
// ======================
async function generateExplanation(careerTitle, matchedDomains, userAnswers) {
  const domains = matchedDomains.map(domainLabel).join(", ");
  
  // Extract meaningful quotes from user answers (actual phrases, not just keywords)
  const getRelevantQuote = (text, maxWords = 8) => {
    if (!text || text.length < 10) return null;
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 5);
    if (sentences.length === 0) return null;
    
    // Get first meaningful sentence
    const quote = sentences[0].trim().split(' ').slice(0, maxWords).join(' ');
    return quote.length > 15 ? quote : null;
  };
  
  const skillQuote = getRelevantQuote(userAnswers.skills);
  const taskQuote = getRelevantQuote(userAnswers.tasks);
  const knowledgeQuote = getRelevantQuote(userAnswers.knowledge);
  
  // Build personalized explanation
  let explanation = "";
  
  // Opening based on match quality
  const matchCount = matchedDomains.length;
  if (matchCount >= 3) {
    explanation += "This is an excellent match for you! ";
  } else if (matchCount === 2) {
    explanation += "This career aligns well with your profile. ";
  } else {
    explanation += "This role could be a good fit. ";
  }
  
  // Connect user's actual words to the career
  const title = careerTitle.toLowerCase();
  
  // Skills connection
  if (matchedDomains.includes("skills") && skillQuote) {
    explanation += `You mentioned that you ${skillQuote.toLowerCase()}, which is exactly the kind of mindset needed in ${careerTitle}. `;
  } else if (matchedDomains.includes("skills")) {
    explanation += `Your natural abilities align with the core competencies required for ${careerTitle}. `;
  }
  
  // Tasks connection  
  if (matchedDomains.includes("tasks") && taskQuote) {
    explanation += `The work you described—${taskQuote.toLowerCase()}—closely mirrors the daily responsibilities in this role. `;
  } else if (matchedDomains.includes("tasks")) {
    explanation += `The type of work you're drawn to matches what professionals in this field do regularly. `;
  }
  
  // Knowledge connection
  if (matchedDomains.includes("knowledge") && knowledgeQuote) {
    explanation += `Your interest in ${knowledgeQuote.toLowerCase()} provides a strong foundation for this career. `;
  } else if (matchedDomains.includes("knowledge")) {
    explanation += `Your educational interests are directly relevant to this profession. `;
  }
  
  // Work environment
  if (matchedDomains.includes("occ")) {
    explanation += `The work environment in this field matches your preferred style. `;
  }
  
  // Career-specific insights and reality checks
  if (title.includes("software") || title.includes("developer") || title.includes("programmer")) {
    explanation += "This field is growing rapidly with excellent job prospects. You'll need to stay current with evolving technologies and be comfortable with continuous learning. Expect to spend significant time debugging code and collaborating with teams.";
    
  } else if (title.includes("data scientist") || title.includes("data analyst")) {
    explanation += "This role combines analytical thinking with business impact. You'll work with large datasets to uncover insights. Strong programming and statistics skills are essential, and you'll often need to explain complex findings to non-technical stakeholders.";
    
  } else if (title.includes("physician") || title.includes("doctor") || title.includes("hospitalist") || title.includes("surgeon")) {
    explanation += "This is a demanding but deeply rewarding career requiring extensive medical education (typically 11+ years including residency). You'll work long hours, make critical decisions under pressure, and directly impact patients' lives. The emotional and intellectual challenges are significant.";
    
  } else if (title.includes("nurse") || title.includes("nursing")) {
    explanation += "This hands-on healthcare role requires strong interpersonal skills and resilience. You'll work closely with patients during vulnerable moments. The work can be physically and emotionally demanding, with shift work common, but offers deep job satisfaction.";
    
  } else if (title.includes("teacher") || title.includes("instructor") || title.includes("educator") || title.includes("professor")) {
    explanation += "Teaching offers the chance to shape young minds and make lasting impact. Beyond classroom time, expect significant prep work, grading, and administrative duties. Patience and adaptability are crucial, as every student learns differently.";
    
  } else if (title.includes("engineer") && !title.includes("software")) {
    explanation += "Engineering careers combine technical problem-solving with practical applications. You'll typically work on projects from concept to completion. Strong math and physics foundations are important, and you'll often collaborate across disciplines.";
    
  } else if (title.includes("manager") || title.includes("director") || title.includes("executive")) {
    explanation += "Leadership roles require balancing people management with strategic thinking. You'll make decisions affecting teams and budgets. Success depends on communication skills, emotional intelligence, and the ability to navigate organizational dynamics.";
    
  } else if (title.includes("designer") || title.includes("ux") || title.includes("ui")) {
    explanation += "Design careers blend creativity with user needs and business goals. You'll iterate based on feedback and data. Building a strong portfolio is essential, and you'll need to stay current with design trends and tools.";
    
  } else if (title.includes("analyst") && !title.includes("data")) {
    explanation += "This analytical role involves examining information to support business decisions. You'll create reports, identify trends, and recommend actions. Strong communication skills are just as important as technical analysis abilities.";
    
  } else if (title.includes("scientist") || title.includes("researcher")) {
    explanation += "Research careers focus on advancing knowledge through systematic investigation. Academic paths typically require a PhD. Expect to write extensively, compete for grants, and work on long-term projects with uncertain outcomes.";
    
  } else if (title.includes("accountant") || title.includes("auditor")) {
    explanation += "This detail-oriented profession requires precision and understanding of financial regulations. While work is often predictable, busy seasons (like tax time) can be intense. Professional certifications like CPA significantly boost career prospects.";
    
  } else if (title.includes("marketing") || title.includes("advertis")) {
    explanation += "Marketing blends creativity with data-driven strategy. You'll need to understand consumer behavior and measure campaign effectiveness. The field evolves quickly with digital channels, requiring continuous learning.";
    
  } else if (title.includes("sales")) {
    explanation += "Sales careers are relationship-driven and results-oriented. Income often includes commission, creating earning potential but also variability. Resilience is crucial—you'll face rejection regularly and need strong interpersonal skills.";
    
  } else if (title.includes("counselor") || title.includes("therapist") || title.includes("psychologist") || title.includes("social worker")) {
    explanation += "This helping profession requires deep empathy and strong boundaries. You'll support people through challenging situations, which can be emotionally taxing. Most roles require specific licensure and ongoing supervision/training.";
    
  } else if (title.includes("lawyer") || title.includes("attorney") || title.includes("legal")) {
    explanation += "Legal careers demand strong analytical and communication skills. Law school and bar exams are rigorous. While portrayed as glamorous, much of the work involves extensive reading, writing, and attention to detail. Hours can be long, especially early in your career.";
    
  } else if (title.includes("writer") || title.includes("author") || title.includes("journalist")) {
    explanation += "Writing professionally requires discipline and thick skin for criticism. Income can be unstable, especially freelance. Success comes from finding your unique voice, meeting deadlines, and constantly improving your craft.";
    
  } else if (title.includes("entrepreneur") || title.includes("founder")) {
    explanation += "Entrepreneurship offers autonomy but comes with significant risk and uncertainty. You'll wear many hats, work long hours, and face frequent setbacks. Financial stability may take years, but the potential for impact and rewards is substantial.";
    
  } else if (title.includes("project manager") || title.includes("program manager")) {
    explanation += "Project management is about coordinating people, timelines, and resources. You'll need strong organizational and communication skills. Success means delivering results while navigating competing priorities and stakeholder expectations.";
    
  } else if (title.includes("human resources") || title.includes("hr")) {
    explanation += "HR balances employee advocacy with business needs. You'll handle confidential matters and navigate interpersonal conflicts. The role requires discretion, empathy, and understanding of employment law.";
    
  } else if (title.includes("financial") || title.includes("finance")) {
    explanation += "Finance roles combine analytical skills with business acumen. You'll work with numbers and models to inform decisions. Professional certifications (CFA, CFP) often boost advancement. The field can be high-pressure, especially in investment banking.";
    
  } else {
    explanation += "This career offers opportunities to grow your expertise over time. Research typical career progression, required education, and work-life balance in this field to ensure it aligns with your long-term goals.";
  }
  
  return explanation;
}

// Optional: If you want to use actual Claude API (requires backend proxy)
async function generateExplanationWithAPI(careerTitle, matchedDomains, userAnswers) {
  const domains = matchedDomains.map(domainLabel).join(", ");
  
  const prompt = `You are a career counselor explaining why a career matches someone's profile.

Career: ${careerTitle}
Matched domains: ${domains}

User's answers:
- Activities they enjoy: "${userAnswers.skills}"
- Subjects that interest them: "${userAnswers.knowledge}"
- Work they find appealing: "${userAnswers.tasks}"
- Preferred work environment: "${userAnswers.occ}"

Provide a brief, friendly explanation (3-4 sentences) of:
1. Why this career matches their interests
2. What specific aspects of their answers align with this role
3. One key thing they should know about this career path

Be encouraging but realistic. Use natural, conversational language.`;

  try {
    // NOTE: This requires a backend proxy to work from browser
    // Direct browser -> Anthropic API calls will fail due to CORS
    const response = await fetch("YOUR_BACKEND_PROXY_URL/api/explain", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt: prompt,
        careerTitle: careerTitle
      })
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    return data.explanation;
  } catch (error) {
    console.error("API explanation error:", error);
    // Fall back to local explanation
    return generateExplanation(careerTitle, matchedDomains, userAnswers);
  }
}

// ======================
// Modal Functions
// ======================
function showModal(title, content) {
  modalTitle.textContent = title;
  modalBody.innerHTML = content;
  modal.classList.add("show");
  document.body.style.overflow = "hidden";
}

function hideModal() {
  modal.classList.remove("show");
  document.body.style.overflow = "";
}

modalClose.addEventListener("click", hideModal);
modal.addEventListener("click", (e) => {
  if (e.target === modal) hideModal();
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && modal.classList.contains("show")) {
    hideModal();
  }
});

async function explainCareer(career) {
  showModal(career.title, `
    <div class="loading-spinner">
      <div class="spinner"></div>
      <p>Generating explanation...</p>
    </div>
  `);

  const explanation = await generateExplanation(
    career.title,
    career.domains,
    currentUserAnswers
  );

  modalBody.innerHTML = `
    <div class="explanation-content">
      <p>${explanation}</p>
      
      <h4>Match Details</h4>
      <ul>
        <li><strong>Matched Domains:</strong> ${career.domains.map(domainLabel).join(", ")}</li>
        <li><strong>Match Strength:</strong> ${(career.avgScore * 100).toFixed(0)}%</li>
        <li><strong>Domains Matched:</strong> ${career.matchCount} out of 4</li>
      </ul>
    </div>
  `;
}

// ======================
// Rendering
// ======================
function renderResults(finalRanked, searchLevel, prefs) {
  resultsEl.innerHTML = "";
  
  // Show results card
  resultsCard.style.display = "block";
  resultsCard.classList.add("show");
  
  // Scroll to results after a short delay
  setTimeout(() => {
    resultsCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, 100);

  // Preference display
  const prefDisplay = document.createElement("div");
  prefDisplay.className = "prefDebug";
  const selectedPrefs = [];
  if (prefs.office) selectedPrefs.push("Office");
  if (prefs.field) selectedPrefs.push("Field");
  if (prefs.industrial) selectedPrefs.push("Industrial");
  if (prefs.military) selectedPrefs.push("Military");
  if (prefs.healthcare) selectedPrefs.push("Healthcare");
  if (prefs.tech) selectedPrefs.push("Tech/IT");
  if (prefs.creative) selectedPrefs.push("Creative");
  if (prefs.education) selectedPrefs.push("Education");
  if (prefs.business) selectedPrefs.push("Business/Finance");
  
  prefDisplay.innerHTML = `
    <strong>Your Preferences:</strong> ${selectedPrefs.length > 0 ? selectedPrefs.join(", ") : "All types (no filter)"}
  `;
  resultsEl.appendChild(prefDisplay);

  if (!finalRanked.length) {
    const noResults = document.createElement("div");
    noResults.className = "meta";
    noResults.innerHTML = `
      <p>We couldn't find strong matches. This could mean:</p>
      <ul style="margin: 8px 0; padding-left: 20px;">
        <li>Your answers are very unique - try related terms</li>
        <li>You selected preferences that filtered out all matches - try checking more boxes</li>
        <li>Try increasing "Results to show" to 30-50</li>
      </ul>`;
    resultsEl.appendChild(noResults);
    return;
  }

  // Quality indicator
  const qualityNote = document.createElement("div");
  qualityNote.className = "searchQuality";
  
  let qualityText = "";
  let qualityClass = "";
  
  if (searchLevel.includes("strict (2+ matches)")) {
    qualityText = "🎯 Strong matches - These careers align well with your interests";
    qualityClass = "quality-high";
  } else if (searchLevel.includes("moderate")) {
    qualityText = "✓ Good matches - These careers have some overlap with your interests";
    qualityClass = "quality-good";
  } else {
    qualityText = "○ Exploratory matches - These careers might be worth considering";
    qualityClass = "quality-moderate";
  }
  
  qualityNote.innerHTML = `<div class="${qualityClass}">${qualityText}</div>`;
  resultsEl.appendChild(qualityNote);

  // Group results
  const grouped = {
    high: finalRanked.filter(r => r.matchCount >= 3),
    medium: finalRanked.filter(r => r.matchCount === 2),
    exploratory: finalRanked.filter(r => r.matchCount === 1)
  };

  const enableExplanations = document.getElementById("enableExplanations")?.checked ?? true;

  // Render high matches
  if (grouped.high.length > 0) {
    const header = document.createElement("h3");
    header.className = "resultGroup";
    header.textContent = "🎯 Top Matches";
    resultsEl.appendChild(header);
    
    grouped.high.slice(0, 5).forEach(r => renderCard(r, prefs, enableExplanations));
  }

  // Render medium matches
  if (grouped.medium.length > 0 && finalRanked.length > 5) {
    const header = document.createElement("h3");
    header.className = "resultGroup";
    header.textContent = "✓ Worth Exploring";
    resultsEl.appendChild(header);
    
    grouped.medium.slice(0, 3).forEach(r => renderCard(r, prefs, enableExplanations));
  }

  // Render exploratory matches
  if (grouped.exploratory.length > 0 && grouped.high.length < 3) {
    const header = document.createElement("h3");
    header.className = "resultGroup";
    header.textContent = "💡 Consider These Too";
    resultsEl.appendChild(header);
    
    grouped.exploratory.slice(0, 3).forEach(r => renderCard(r, prefs, enableExplanations));
  }
  
  console.log(`Rendered ${finalRanked.length} results in ${resultsEl.children.length} elements`);
}

function renderCard(r, prefs, enableExplanations) {
  const domainsNice = r.domains.map(domainLabel).join(", ");
  
  let badgeClass = "badge";
  if (r.matchCount >= 3) badgeClass += " badge-good";
  else if (r.matchCount >= 2) badgeClass += " badge-moderate";
  
  const cls = classifyTitle(r.title);
  const roleType = cls.isMilitary ? "Military" : 
                   cls.isHealthcare ? "Healthcare" :
                   cls.isTech ? "Tech/IT" :
                   cls.isCreative ? "Creative" :
                   cls.isEducation ? "Education" :
                   cls.isBusiness ? "Business/Finance" :
                   cls.isIndustrial ? "Industrial" :
                   cls.isField ? "Field" : "Office";
  
  let boostDetails = "";
  if (r.appliedBoost && Math.abs(r.appliedBoost) > 0.05) {
    boostDetails = ` • <strong>${r.appliedBoost > 0 ? 'Applied' : 'Theory'} role:</strong> ${r.appliedBoost > 0 ? '+' : ''}${r.appliedBoost.toFixed(2)}`;
  }
  
  // Encode job title for search URLs
  const encodedTitle = encodeURIComponent(r.title);
  const linkedInUrl = `https://www.linkedin.com/jobs/search/?keywords=${encodedTitle}`;
  const indeedUrl = `https://www.indeed.com/jobs?q=${encodedTitle}`;
  const googleJobsUrl = `https://www.google.com/search?q=${encodedTitle}+jobs`;
  
  const el = document.createElement("div");
  el.className = "resultCard";
  el.innerHTML = `
    <div class="resultTop">
      <div class="resultTitle">${r.title}</div>
      <div class="${badgeClass}">${r.matchCount}/4</div>
    </div>
    <div class="meta">
      <strong>Matched:</strong> ${domainsNice} • 
      <strong>Type:</strong> ${roleType} • 
      <strong>Match strength:</strong> ${(r.avgScore * 100).toFixed(0)}%${boostDetails}
    </div>
    <div class="card-actions">
      ${enableExplanations ? `
        <button class="explain-btn" data-career-id="${r.id}">
          <span>🤖</span>
          <span>Why this matches</span>
        </button>
      ` : ''}
      
      <div class="job-search-dropdown">
        <button class="linkedin-btn dropdown-toggle">
          <span>💼</span>
          <span>Find Opportunities</span>
          <span class="dropdown-arrow">▼</span>
        </button>
        <div class="dropdown-menu">
          <a href="${linkedInUrl}" target="_blank" rel="noopener noreferrer" class="dropdown-item">
            <span class="dropdown-icon">💼</span>
            <span>LinkedIn Jobs</span>
          </a>
          <a href="${indeedUrl}" target="_blank" rel="noopener noreferrer" class="dropdown-item">
            <span class="dropdown-icon">🔍</span>
            <span>Indeed</span>
          </a>
          <a href="${googleJobsUrl}" target="_blank" rel="noopener noreferrer" class="dropdown-item">
            <span class="dropdown-icon">🌐</span>
            <span>Google Jobs</span>
          </a>
        </div>
      </div>
    </div>
  `;
  
  if (enableExplanations) {
    const explainBtn = el.querySelector(".explain-btn");
    explainBtn.addEventListener("click", () => explainCareer(r));
  }
  
  // Setup dropdown toggle
  const dropdown = el.querySelector(".job-search-dropdown");
  const toggle = el.querySelector(".dropdown-toggle");
  const menu = el.querySelector(".dropdown-menu");
  
  toggle.addEventListener("click", (e) => {
    e.stopPropagation();
    
    // Close other dropdowns
    document.querySelectorAll(".job-search-dropdown.active").forEach(d => {
      if (d !== dropdown) d.classList.remove("active");
    });
    
    dropdown.classList.toggle("active");
  });
  
  // Close dropdown when clicking outside
  document.addEventListener("click", () => {
    dropdown.classList.remove("active");
  });
  
  resultsEl.appendChild(el);
}

// ======================
// Export Functionality
// ======================
exportBtn.addEventListener("click", () => {
  const resultsText = Array.from(document.querySelectorAll(".resultCard"))
    .map((card, i) => {
      const title = card.querySelector(".resultTitle").textContent;
      const meta = card.querySelector(".meta").textContent;
      return `${i + 1}. ${title}\n   ${meta}\n`;
    })
    .join("\n");

  const blob = new Blob([resultsText], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "career-matches.txt";
  a.click();
  URL.revokeObjectURL(url);
});

// ======================
// Init
// ======================
async function init() {
  runBtn.disabled = true;
  showProgress("Loading AI model...", 10);

  try {
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
    showProgress("Loading career database...", 50);

    const domains = ["skills", "knowledge", "tasks", "occ"];
    store = {};
    
    for (let i = 0; i < domains.length; i++) {
      const d = domains[i];
      showProgress(`Loading ${d}...`, 50 + (i / domains.length) * 40);
      const [meta, npy] = await Promise.all([
        loadJSON(`${EMB_DIR}/${d}_meta.json`),
        loadNPY(`${EMB_DIR}/${d}.npy`),
      ]);
      store[d] = { meta, npy };
    }

    showProgress("Ready! ✨", 100);
    setTimeout(() => {
      hideProgress();
    }, 500);
    
    runBtn.disabled = false;
  } catch (error) {
    console.error("Initialization error:", error);
    setStatus("Error loading. Please refresh the page.");
  }
}

init();

// ======================
// Show More Preferences Toggle
// ======================
const showMoreBtn = document.getElementById("showMoreBtn");
const preferencesSection = document.querySelector(".preferences-section");

if (showMoreBtn && preferencesSection) {
  const btnText = showMoreBtn.querySelector("span:first-child");
  
  showMoreBtn.addEventListener("click", () => {
    preferencesSection.classList.toggle("expanded");
    
    // Update button text
    if (preferencesSection.classList.contains("expanded")) {
      btnText.textContent = "Show Less";
    } else {
      btnText.textContent = "Show More";
    }
  });
}

// ======================
// Main Form Handler
// ======================
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!store || !embedder) return;

  const answers = {
    skills: document.getElementById("q1").value.trim(),
    knowledge: document.getElementById("q2").value.trim(),
    tasks: document.getElementById("q3").value.trim(),
    occ: document.getElementById("q4").value.trim(),
  };

  currentUserAnswers = answers; // Store for explanations

  const prefs = readPreferences();
  const topK = Math.max(5, Math.min(50, Number(document.getElementById("topK").value || 20)));

  const totalLength = Object.values(answers).reduce((sum, v) => sum + v.length, 0);
  if (totalLength < 20) {
    setStatus("Please write at least a few words in each box to help us understand your interests.");
    return;
  }

  runBtn.disabled = true;
  resultsEl.innerHTML = "";
  resultsCard.style.display = "none";
  resultsCard.classList.remove("show");
  setStatus("🧠 Understanding your interests...");

  try {
    // Embed answers
    const userVecs = {};
    for (const [domain, text] of Object.entries(answers)) {
      setStatus(`🔍 Analyzing your ${domain}...`);
      userVecs[domain] = await embedText(text);
    }

    // Adaptive search
    setStatus(`🎯 Finding matching careers...`);
    let results = performSearch(userVecs, SIM_THRESHOLD_STRICT, MIN_MATCHES_STRICT, topK);
    let searchLevel = "strict (2+ matches)";

    if (results.length < 5) {
      results = performSearch(userVecs, SIM_THRESHOLD_RELAXED, MIN_MATCHES_NORMAL, topK);
      searchLevel = "moderate (1+ match)";
    }

    if (results.length < 5) {
      results = performSearch(userVecs, SIM_THRESHOLD_MIN, MIN_MATCHES_RELAXED, topK);
      searchLevel = "exploratory (1+ match)";
    }

    // Apply diversity scoring
    results = calculateDiversity(results);

    // Apply preferences and rank
    const finalRanked = results
      .map((r) => {
        const avgScore = r.scores.reduce((a, b) => a + b, 0) / r.scores.length;
        const pref = preferenceBoost(r.title, prefs);
        const applied = getAppliedBoost(r.title, answers);
        const finalScore = r.weightedScore + pref + applied + (r.diversityScore * 0.01);

        return { ...r, avgScore, prefBoost: pref, appliedBoost: applied, finalScore };
      })
      .sort((a, b) => {
        if (b.matchCount !== a.matchCount) return b.matchCount - a.matchCount;
        if (Math.abs(b.finalScore - a.finalScore) > 0.01) return b.finalScore - a.finalScore;
        return b.avgScore - a.avgScore;
      });

    setStatus(`✨ Found ${finalRanked.length} career options for you!`);
    console.log("Final ranked results:", finalRanked);
    console.log("Search level:", searchLevel);
    console.log("Preferences:", prefs);
    renderResults(finalRanked, searchLevel, prefs);
  } catch (err) {
    console.error(err);
    setStatus("❌ Error analyzing your answers. Please try again.");
  } finally {
    runBtn.disabled = false;
  }
});