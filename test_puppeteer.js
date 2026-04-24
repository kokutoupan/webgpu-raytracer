import puppeteer from 'puppeteer';

(async () => {
  const browser = await puppeteer.launch({
    ignoreHTTPSErrors: true,
    args: ['--enable-unsafe-webgpu']
  });
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
  page.on('requestfailed', request => console.log('REQUEST FAILED:', request.url(), request.failure().errorText));

  await page.goto('https://localhost:5174/webgpu-raytracer/');
  
  // Wait for button and click
  await page.waitForSelector('button#render-btn', { timeout: 10000 }).catch(e => console.log("Button not found:", e.message));
  
  console.log("Clicking render start...");
  await page.evaluate(() => {
    const btn = Array.from(document.querySelectorAll('button')).find(el => el.textContent === 'Render start');
    if(btn) btn.click();
    else {
      const renderBtn = document.getElementById('render-btn');
      if(renderBtn) renderBtn.click();
    }
  });

  await new Promise(r => setTimeout(r, 5000));
  
  await browser.close();
})();
