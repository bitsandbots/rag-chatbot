// Drives the rag-chatbot /chat UI end-to-end via headless Chromium.
// Usage: PORT=5099 node check.js  (PORT defaults to 5000)
const { chromium } = require('playwright-core');

const port = process.env.PORT || '5000';
const baseUrl = `http://localhost:${port}`;

(async () => {
  const browser = await chromium.launch({
    executablePath: '/usr/bin/chromium',
    args: ['--no-sandbox'],
  });
  const page = await browser.newPage();
  const consoleErrors = [];
  page.on('console', (msg) => {
    console.log('CONSOLE:', msg.type(), msg.text());
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });
  page.on('pageerror', (err) => consoleErrors.push(String(err)));
  page.on('requestfailed', (req) => console.log('REQUEST_FAILED:', req.url(), req.failure()));

  await page.goto(`${baseUrl}/chat`, { waitUntil: 'networkidle' });
  await page.screenshot({ path: `${__dirname}/1-loaded.png` });

  const title = await page.textContent('header h1');
  console.log('HEADER:', title.trim());

  await page.waitForSelector('#input-bar button', { state: 'visible' });
  await page.fill('#question', 'What is this project about?');
  await page.click('#input-bar button');

  await page.waitForSelector('.msg.assistant', { timeout: 180000 });
  await page.screenshot({ path: `${__dirname}/2-response.png` });

  const messages = await page.$$eval('.msg', (els) =>
    els.map((e) => ({ cls: e.className, text: e.textContent })),
  );
  console.log('MESSAGES:', JSON.stringify(messages, null, 2));
  console.log('CONSOLE_ERRORS:', JSON.stringify(consoleErrors));

  await browser.close();
})().catch((err) => {
  console.error('DRIVER_ERROR:', err);
  process.exit(1);
});
