#!/usr/bin/env node
/**
 * rerun_axe_failures.js
 *
 * Usage:
 *   node rerun_axe_failures.js failures.txt
 *
 * Reads each line of failures.txt of the form:
 *   1656258638044@5<TAB>file:///…/iPhone-13 Pro-html.html<TAB>Error message
 * and retries Axe with:
 *   • waitUntil: 'domcontentloaded'
 *   • blocking external resources
 *   • scoping to <html>
 * Writes updated -axe.json in place, and logs any still‑failing cases
 * immediately to rerun_failures_remaining.txt.
 */

const fs        = require('fs');
const path      = require('path');
const puppeteer = require('puppeteer');
const { AxePuppeteer } = require('axe-puppeteer');

// helper to append a single line to the “remaining” file
function logRemaining(line) {
  try {
    fs.appendFileSync('rerun_failures_remaining.txt', line + '\n');
  } catch (e) {
    console.error('   ❌ Could not write to rerun_failures_remaining.txt:', e.message);
  }
}

;(async () => {
  const [, , failuresFile] = process.argv;
  if (!failuresFile) {
    console.error('Usage: node rerun_axe_failures.js <failures.txt>');
    process.exit(1);
  }

  // start with a fresh file
  try {
    fs.writeFileSync('rerun_failures_remaining.txt', '');
  } catch (e) {
    console.error('❌ Could not initialize rerun_failures_remaining.txt:', e.message);
    process.exit(1);
  }

  // read and sanitize lines
  let lines;
  try {
    lines = fs.readFileSync(failuresFile, 'utf-8')
      .split('\n')
      .map(l => l.trim())
      .filter(l => l.length > 0);
  } catch (e) {
    console.error('❌ Could not read failures file:', e.message);
    process.exit(1);
  }

  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox','--disable-dev-shm-usage'],
  });
  const page = await browser.newPage();

  // block heavy/remote resources
  await page.setRequestInterception(true);
  page.on('request', req => {
    const t = req.resourceType();
    if (['image','media','font','stylesheet','script'].includes(t)) req.abort();
    else req.continue();
  });

  for (const line of lines) {
    const parts = line.split('\t');
    const combo   = parts[0];
    const htmlUrl = parts[1];
    console.log(`\n→ Retrying ${combo}`);

    try {
      // derive local filesystem path
      let htmlPath;
      try {
        const urlObj    = new URL(htmlUrl);
        htmlPath        = decodeURIComponent(urlObj.pathname);
      } catch (e) {
        throw new Error(`Bad URL "${htmlUrl}": ${e.message}`);
      }

      const dir      = path.dirname(htmlPath);
      const htmlName = path.basename(htmlPath);
      const prefix   = htmlName.replace(/-html\.html$/, '');
      const outFile  = path.join(dir, `${prefix}-axe.json`);

      // ensure output directory exists
      fs.mkdirSync(dir, { recursive: true });

      // 1) load page
      try {
        await page.goto(htmlUrl, {
          waitUntil: 'domcontentloaded',
          timeout: 60000,
        });
      } catch (gotoErr) {
        console.error(`   ❌ load error: ${gotoErr.message}`);
        logRemaining(`${line}\t${gotoErr.message}`);
        continue;
      }

      // 1b) wait for <html> to actually be there
      try {
        await page.waitForSelector('html', { timeout: 30000 });
      } catch (selErr) {
        console.error(`   ❌ no <html> found: ${selErr.message}`);
        logRemaining(`${line}\tNo <html> element: ${selErr.message}`);
        continue;
      }

      // 2) run Axe scoped to html
      let results;
      try {
        results = await new AxePuppeteer(page)
          .include('html')
          .analyze();
      } catch (axeErr) {
        console.error(`   ⚠️  Axe still failing: ${axeErr.message}`);
        // stub out an empty violations file so later phases don't break
        const stub = { error: axeErr.message, violations: [] };
        try {
          fs.writeFileSync(outFile, JSON.stringify(stub, null, 2));
        } catch (writeErr) {
          console.error(`   ❌ Could not write stub to ${outFile}: ${writeErr.message}`);
        }
        logRemaining(`${line}\t${axeErr.message}`);
        continue;
      }

      // success → write real results
      try {
        fs.writeFileSync(outFile, JSON.stringify(results, null, 2));
        console.log(`   ✅ Success, wrote ${outFile}`);
      } catch (writeErr) {
        console.error(`   ❌ Could not write results to ${outFile}: ${writeErr.message}`);
        logRemaining(`${line}\tWrite error: ${writeErr.message}`);
      }

    } catch (fatal) {
      console.error(`   🔥 Fatal processing error: ${fatal.message}`);
      logRemaining(`${line}\t${fatal.message}`);
    }
  }

  await browser.close();
  console.log('\n🏁 Retry pass complete. Check rerun_failures_remaining.txt for any still‑failing items.');
})();
