#!/usr/bin/env node
const fs        = require('fs');
const path      = require('path');
const puppeteer = require('puppeteer');
const { AxePuppeteer } = require('axe-puppeteer');

const FAILURE_LOG = '/Users/akshat/Data/UIUC/Spring 2025/Courses/CS 568 User-Centered Machine Learning/Project/WebUI-7k/axe_failures.txt';

// clear out old failure log
try {
  fs.writeFileSync(FAILURE_LOG, '');
} catch {
  // ignore
}

(async () => {
  const [, , jobsFile] = process.argv;
  if (!jobsFile) {
    console.error('Usage: node run-axe-puppeteer.js <axe_jobs.json>');
    process.exit(1);
  }

  let jobs;
  try {
    jobs = JSON.parse(fs.readFileSync(jobsFile, 'utf-8'));
  } catch (err) {
    console.error('❌ Failed to read jobs file:', err.message);
    process.exit(1);
  }

  const browser = await puppeteer.launch({
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
    headless: true,
  });
  const page = await browser.newPage();

  await page.setRequestInterception(true);
  page.on('request', req => {
    const t = req.resourceType();
    if (['image','media','font','stylesheet'].includes(t)) req.abort();
    else req.continue();
  });

  for (const { htmlUrl, outFile, pageId, vpIndex } of jobs) {
    console.log(`→ [${pageId}@${vpIndex}] ⏳ loading ${htmlUrl}`);
    try {
      await page.goto(htmlUrl, {
        waitUntil: 'networkidle2',
        timeout: 30000,
      });

      console.log(`→ [${pageId}@${vpIndex}] 🪝 running axe`);
      const results = await new AxePuppeteer(page)
        .include('body')
        .analyze();

      fs.writeFileSync(outFile, JSON.stringify(results, null, 2));
      console.log(`→ [${pageId}@${vpIndex}] ✅ done`);
    } catch (err) {
      console.error(`⚠️  Axe failed for ${pageId}@${vpIndex}:`, err.message);
      const stub = { error: err.message, violations: [] };
      try {
        fs.writeFileSync(outFile, JSON.stringify(stub, null, 2));
      } catch (writeErr) {
        console.error(`❌ Could not write stub for ${pageId}@${vpIndex}:`, writeErr.message);
      }
      try {
        fs.appendFileSync(
          FAILURE_LOG,
          `${pageId}@${vpIndex}\t${htmlUrl}\t${err.message}\n`
        );
      } catch (logErr) {
        console.error(`❌ Could not log failure for ${pageId}@${vpIndex}:`, logErr.message);
      }
    }
  }

  await browser.close();
  console.log('🏁 All Axe jobs complete.');
})();
