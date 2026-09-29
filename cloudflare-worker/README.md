# AI assistant (Cloudflare Worker)

`worker.js` connects the site's "Ask a question" chat to Groq. The Groq key is stored as a Cloudflare secret and never appears in this repo or in the browser.

1. Cloudflare dashboard > Workers & Pages > Create > Create Worker. Name it `mah-assistant`, then Deploy.
2. Edit code: delete everything, paste the contents of `worker.js`, then Deploy.
3. Settings > Variables and Secrets > Add: type **Secret**, name `GROQ_API_KEY`, value = your Groq key. Deploy.
4. Copy the Worker's URL (for example `https://mah-assistant.<your-subdomain>.workers.dev`) and set it as `AI_URL` at the top of `assets/js/assistant.js`.

If the Worker is down or Groq's free quota runs out, the chat automatically falls back to its built-in answers.
