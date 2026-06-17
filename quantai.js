// QuantAI Interface - Advanced Conversational Engine
// © Model Analysis Hub | Built by Amadu Kamal

(function () {
  'use strict';

  // ─── Knowledge Base ───────────────────────────────────────────────────────
  const KB = {
    founder: {
      name: 'Amadu Kamal',
      title: 'Software Engineer, Data Scientist & Founder',
      org: 'Model Analysis Hub',
      age: 'young innovator redefining the tech landscape in Africa and beyond',
      bio: `Amadu Kamal is a visionary Software Engineer and Data Scientist who sits at the 
            vanguard of predictive analytics and enterprise software architecture. He founded 
            Model Analysis Hub to bridge the gap between rigorous statistical research and 
            cutting-edge digital product development. His work spans geospatial disease modeling, 
            full-stack web platforms, and applied AI systems.`,
      github: 'https://github.com/kamaldeenjnr',
      linkedin: 'https://www.linkedin.com/in/amadu-kamal-65ab5326a',
      email: 'amadukamal8@gmail.com',
      phone: '+233 595 586 430',
      whatsapp: 'https://wa.me/233595586430',
      location: 'Ghana, West Africa',
      education: 'Statistical Modeling, Computer Science & Geospatial Systems',
    },
    quantai: {
      name: 'QuantAI',
      description: `QuantAI is a high-performance computational intelligence engine built under 
                    Model Analysis Hub. It is designed to process complex unstructured data, 
                    execute advanced predictive models, and deliver enterprise-grade analytical 
                    intelligence via a clean API-first architecture.`,
      status: 'Currently in preview - full launch coming soon.',
      features: ['Bayesian predictive modeling', 'Real-time data stream processing', 
                 'Natural language querying of datasets', 'Automated model validation', 
                 'Scalable REST API integration'],
    },
    projects: [
      {
        title: 'Malaria Risk Prediction Model - Ghana',
        type: 'Geospatial Analysis',
        stack: ['Python', 'QGIS', 'GeoPandas', 'R'],
        desc: `An advanced spatial epidemiological model that integrates climate variables, 
               population density, and historical case data to map and forecast malaria 
               outbreak probability across Ghanaian regions.`,
      },
      {
        title: 'Model Validation Framework',
        type: 'Statistical Analytics',
        stack: ['R', 'Statistics', 'Data Validation'],
        desc: `A rigorous observed-vs-predicted comparative analysis framework that validates 
               the accuracy of deployed predictive models using residual diagnostics and 
               confidence interval mapping.`,
      },
      {
        title: 'Risk Modeling Analytics',
        type: 'Risk Modeling',
        stack: ['Python', 'Probabilistic Math', 'NumPy'],
        desc: `Probabilistic modeling system that calculates the likelihood of disease case 
               loads exceeding critical thresholds, enabling proactive resource deployment 
               and emergency preparedness planning.`,
      },
      {
        title: 'Enterprise Analytics Dashboard',
        type: 'Full-Stack Web Development',
        stack: ['JavaScript', 'Tailwind CSS', 'REST API'],
        desc: `A full-stack business intelligence portal engineered for real-time streaming 
               data visualization, KPI tracking, and automated report generation for 
               enterprise clients.`,
      },
    ],
    services: [
      'Data Analytics & Statistical Modeling',
      'Geospatial Analysis & Mapping',
      'AI & Predictive System Development',
      'Full-Stack Web Application Development',
      'Business Intelligence Dashboard Design',
    ],
  };

  // ─── Intent Detection ──────────────────────────────────────────────────────
  const intents = [
    {
      patterns: ['who is amadu', 'tell me about amadu', 'about the founder', 'who built this', 'who are you', 'founder', 'owner'],
      respond: () => `<strong>${KB.founder.name}</strong> is a ${KB.founder.age}.<br><br>
        ${KB.founder.bio}<br><br>
        <a href="founder.html" class="qai-link">View Full Profile →</a>`,
    },
    {
      patterns: ['contact', 'phone', 'number', 'reach', 'email', 'whatsapp', 'call', 'hire', 'get in touch'],
      respond: () => `You can reach <strong>Amadu Kamal</strong> directly:<br><br>
        📞 &nbsp;<strong>${KB.founder.phone}</strong><br>
        ✉️ &nbsp;<a href="mailto:${KB.founder.email}" class="qai-link">${KB.founder.email}</a><br>
        💼 &nbsp;<a href="${KB.founder.linkedin}" target="_blank" class="qai-link">LinkedIn Profile</a><br>
        💬 &nbsp;<a href="${KB.founder.whatsapp}" target="_blank" class="qai-link">WhatsApp Direct</a><br>
        🐙 &nbsp;<a href="${KB.founder.github}" target="_blank" class="qai-link">GitHub Repository</a>`,
    },
    {
      patterns: ['project', 'work', 'portfolio', 'analysis', 'dashboard', 'built', 'paper', 'research', 'study'],
      respond: () => {
        const list = KB.projects.map(p =>
          `<div class="qai-proj">📌 <strong>${p.title}</strong> <span class="qai-tag">${p.type}</span></div>`
        ).join('');
        return `Pulling project records...<br><br>${list}<br>
          <a href="projects.html" class="qai-link">Explore Full Portfolio →</a>`;
      },
    },
    {
      patterns: ['quantai', 'what is quantai', 'ai platform', 'ai system', 'quantai features', 'the ai'],
      respond: () => {
        const features = KB.quantai.features.map(f => `• ${f}`).join('<br>');
        return `<strong>QuantAI</strong> - ${KB.quantai.description}<br><br>
          <strong>Core Capabilities:</strong><br>${features}<br><br>
          Status: <span style="color:#4ade80">${KB.quantai.status}</span><br>
          <a href="ai.html" class="qai-link">Learn More About QuantAI →</a>`;
      },
    },
    {
      patterns: ['service', 'what do you do', 'what do you offer', 'offer', 'capabilities', 'specialise', 'specialize'],
      respond: () => {
        const list = KB.services.map(s => `• ${s}`).join('<br>');
        return `<strong>Model Analysis Hub</strong> delivers the following expert services:<br><br>${list}<br><br>
          <a href="mailto:${KB.founder.email}" class="qai-link">Request a Consultation →</a>`;
      },
    },
    {
      patterns: ['location', 'where', 'country', 'based', 'ghana', 'africa'],
      respond: () => `Model Analysis Hub is based in <strong>${KB.founder.location}</strong>, with the capability and vision to serve international clients globally. Amadu's work in geospatial modeling and AI places him at the forefront of African tech innovation.`,
    },
    {
      patterns: ['help', 'what can you do', 'commands', 'options', 'menu'],
      respond: () => `I am the QuantAI interface. I can assist you with:<br><br>
        • <strong>About Amadu</strong> - Tell me about the founder<br>
        • <strong>Contact</strong> - Get direct contact details<br>
        • <strong>Projects</strong> - Browse the project portfolio<br>
        • <strong>QuantAI</strong> - Explore the AI platform<br>
        • <strong>Services</strong> - View what we offer<br>
        • <strong>Location</strong> - Where we are based`,
    },
  ];

  function getResponse(input) {
    const lower = input.toLowerCase().trim();
    for (const intent of intents) {
      if (intent.patterns.some(p => lower.includes(p))) {
        return intent.respond();
      }
    }
    return `I did not find a direct match for "<em>${input}</em>". Try asking about:<br><br>
      <strong>Amadu</strong> · <strong>Contact</strong> · <strong>Projects</strong> · <strong>QuantAI</strong> · <strong>Services</strong>`;
  }

  // ─── Render UI ─────────────────────────────────────────────────────────────
  const CSS = `
    #qai-widget * { box-sizing: border-box; font-family: 'Inter', system-ui, sans-serif; }
    #qai-btn {
      position: fixed; bottom: 28px; right: 28px; z-index: 9999;
      width: 60px; height: 60px; border-radius: 50%;
      background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
      border: none; cursor: pointer; color: #fff; font-size: 22px;
      box-shadow: 0 8px 30px rgba(37,99,235,0.45);
      display: flex; align-items: center; justify-content: center;
      transition: transform 0.25s, box-shadow 0.25s;
    }
    #qai-btn:hover { transform: scale(1.1); box-shadow: 0 12px 40px rgba(37,99,235,0.55); }
    #qai-window {
      position: fixed; bottom: 100px; right: 28px; z-index: 9998;
      width: 370px; max-width: calc(100vw - 40px);
      border-radius: 20px; overflow: hidden;
      box-shadow: 0 25px 80px rgba(0,0,0,0.18);
      border: 1px solid rgba(255,255,255,0.9);
      background: rgba(255,255,255,0.97);
      backdrop-filter: blur(20px);
      transform-origin: bottom right;
      transform: scale(0.9) translateY(20px);
      opacity: 0; pointer-events: none;
      transition: transform 0.3s cubic-bezier(0.34,1.56,0.64,1), opacity 0.25s ease;
    }
    #qai-window.open { transform: scale(1) translateY(0); opacity: 1; pointer-events: all; }
    .qai-header {
      background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
      padding: 16px 18px; color: #fff;
      display: flex; align-items: center; justify-content: space-between;
    }
    .qai-header-left { display: flex; align-items: center; gap: 10px; }
    .qai-status { width: 8px; height: 8px; border-radius: 50%; background: #4ade80; animation: qpulse 2s infinite; }
    @keyframes qpulse { 0%,100%{opacity:1}50%{opacity:0.4} }
    .qai-title { font-weight: 700; font-size: 14px; letter-spacing: 0.5px; }
    .qai-subtitle { font-size: 11px; opacity: 0.75; }
    .qai-close { background: none; border: none; color: rgba(255,255,255,0.7); cursor: pointer; font-size: 16px; padding: 4px; line-height: 1; }
    .qai-close:hover { color: #fff; }
    #qai-body {
      height: 320px; overflow-y: auto; padding: 16px;
      display: flex; flex-direction: column; gap: 12px;
      background: #f8fafc;
    }
    #qai-body::-webkit-scrollbar { width: 4px; }
    #qai-body::-webkit-scrollbar-track { background: transparent; }
    #qai-body::-webkit-scrollbar-thumb { background: rgba(37,99,235,0.2); border-radius: 4px; }
    .qai-msg { display: flex; gap: 8px; align-items: flex-start; animation: qfadeup 0.3s ease; }
    @keyframes qfadeup { from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)} }
    .qai-msg.user { flex-direction: row-reverse; }
    .qai-avatar {
      width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
      background: linear-gradient(135deg, #dbeafe, #ede9fe);
      display: flex; align-items: center; justify-content: center;
      color: #4f46e5; font-size: 13px;
    }
    .qai-bubble {
      padding: 10px 14px; border-radius: 16px; max-width: 260px;
      font-size: 13px; line-height: 1.6; color: #1e293b;
      background: #fff; border: 1px solid #e2e8f0;
      box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .qai-msg.user .qai-bubble {
      background: linear-gradient(135deg, #2563eb, #7c3aed);
      color: #fff; border: none; box-shadow: 0 4px 12px rgba(37,99,235,0.3);
      border-radius: 16px 16px 4px 16px;
    }
    .qai-msg.bot .qai-bubble { border-radius: 16px 16px 16px 4px; }
    .qai-link { color: #2563eb; text-decoration: underline; font-weight: 600; }
    .qai-proj { margin: 4px 0; font-size: 12px; }
    .qai-tag { 
      display: inline-block; font-size: 10px; font-weight: 600; padding: 2px 6px;
      background: #ede9fe; color: #7c3aed; border-radius: 99px; margin-left: 4px;
    }
    .qai-typing { display: flex; gap: 5px; align-items: center; padding: 12px 14px; }
    .qai-dot { width: 7px; height: 7px; border-radius: 50%; background: #94a3b8; animation: qdot 1s infinite; }
    .qai-dot:nth-child(2){animation-delay:0.15s}
    .qai-dot:nth-child(3){animation-delay:0.3s}
    @keyframes qdot { 0%,80%,100%{transform:scale(1);opacity:0.4}40%{transform:scale(1.3);opacity:1} }
    .qai-footer { padding: 12px 14px; background: #fff; border-top: 1px solid #e2e8f0; }
    .qai-form { display: flex; gap: 8px; }
    .qai-input {
      flex: 1; padding: 10px 14px; border-radius: 12px; border: 1.5px solid #e2e8f0;
      font-size: 13px; outline: none; transition: border-color 0.2s;
      background: #f8fafc; color: #1e293b;
    }
    .qai-input:focus { border-color: #2563eb; background: #fff; }
    .qai-send {
      width: 40px; height: 40px; border-radius: 12px; border: none; cursor: pointer;
      background: linear-gradient(135deg, #2563eb, #7c3aed); color: #fff; font-size: 14px;
      display: flex; align-items: center; justify-content: center;
      transition: opacity 0.2s;
    }
    .qai-send:hover { opacity: 0.85; }
    .qai-suggestions { display: flex; flex-wrap: wrap; gap: 6px; padding: 8px 14px 4px; }
    .qai-chip {
      padding: 5px 12px; border-radius: 99px; border: 1.5px solid #e2e8f0;
      background: #fff; font-size: 11px; font-weight: 600; color: #64748b;
      cursor: pointer; transition: all 0.2s;
    }
    .qai-chip:hover { border-color: #2563eb; color: #2563eb; background: #eff6ff; }
  `;

  const HTML = `
    <style>${CSS}</style>
    <div id="qai-widget">
      <button id="qai-btn" aria-label="Open QuantAI Chat">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 2a8 8 0 0 1 8 8v4a8 8 0 0 1-8 8H5l-3 3V10A8 8 0 0 1 12 2z"/>
          <circle cx="8.5" cy="11" r="1.25" fill="currentColor"/>
          <circle cx="12" cy="11" r="1.25" fill="currentColor"/>
          <circle cx="15.5" cy="11" r="1.25" fill="currentColor"/>
        </svg>
      </button>
      <div id="qai-window">
        <div class="qai-header">
          <div class="qai-header-left">
            <div class="qai-status"></div>
            <div>
              <div class="qai-title">QuantAI Interface</div>
              <div class="qai-subtitle">Model Analysis Hub</div>
            </div>
          </div>
          <button class="qai-close" id="qai-close">✕</button>
        </div>
        <div id="qai-body"></div>
        <div class="qai-suggestions">
          <button class="qai-chip" data-q="Who is Amadu?">Who is Amadu?</button>
          <button class="qai-chip" data-q="Show projects">Projects</button>
          <button class="qai-chip" data-q="Contact details">Contact</button>
          <button class="qai-chip" data-q="What is QuantAI?">QuantAI</button>
        </div>
        <div class="qai-footer">
          <form class="qai-form" id="qai-form">
            <input class="qai-input" id="qai-input" type="text" placeholder="Ask anything about MAH..." autocomplete="off">
            <button class="qai-send" type="submit">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
            </button>
          </form>
        </div>
      </div>
    </div>
  `;

  // ─── Mount ─────────────────────────────────────────────────────────────────
  document.body.insertAdjacentHTML('beforeend', HTML);

  const btn = document.getElementById('qai-btn');
  const win = document.getElementById('qai-window');
  const body = document.getElementById('qai-body');
  const form = document.getElementById('qai-form');
  const input = document.getElementById('qai-input');
  const closeBtn = document.getElementById('qai-close');
  let isOpen = false;

  function toggleWindow() {
    isOpen = !isOpen;
    win.classList.toggle('open', isOpen);
    if (isOpen && body.children.length === 0) {
      addBotMsg(`Welcome to <strong>Model Analysis Hub</strong>.<br>I am the QuantAI interface - your gateway to everything about our platform, our founder, and our work.<br><br>What can I help you with today?`);
    }
    if (isOpen) input.focus();
  }

  btn.addEventListener('click', toggleWindow);
  closeBtn.addEventListener('click', () => { isOpen = false; win.classList.remove('open'); });

  // Chip shortcuts
  document.querySelectorAll('.qai-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const q = chip.getAttribute('data-q');
      input.value = q;
      sendMessage(q);
    });
  });

  function addBotMsg(html) {
    const el = document.createElement('div');
    el.className = 'qai-msg bot';
    el.innerHTML = `<div class="qai-avatar">Q</div><div class="qai-bubble">${html}</div>`;
    body.appendChild(el);
    body.scrollTop = body.scrollHeight;
  }

  function addUserMsg(text) {
    const el = document.createElement('div');
    el.className = 'qai-msg user';
    el.innerHTML = `<div class="qai-bubble">${text}</div>`;
    body.appendChild(el);
    body.scrollTop = body.scrollHeight;
  }

  function showTyping() {
    const el = document.createElement('div');
    el.className = 'qai-msg bot'; el.id = 'qai-typing';
    el.innerHTML = `<div class="qai-avatar">Q</div><div class="qai-bubble"><div class="qai-typing"><div class="qai-dot"></div><div class="qai-dot"></div><div class="qai-dot"></div></div></div>`;
    body.appendChild(el);
    body.scrollTop = body.scrollHeight;
  }

  function removeTyping() {
    const t = document.getElementById('qai-typing');
    if (t) t.remove();
  }

  function sendMessage(text) {
    if (!text.trim()) return;
    input.value = '';
    addUserMsg(text);
    showTyping();
    setTimeout(() => {
      removeTyping();
      addBotMsg(getResponse(text));
    }, 700 + Math.random() * 300);
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    sendMessage(input.value.trim());
  });
})();
