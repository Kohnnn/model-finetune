const { chromium } = require('playwright');

async function extractImages(url, blogName) {
    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    
    console.log(`\n=== ${blogName} ===`);
    console.log(`URL: ${url}`);
    
    try {
        await page.goto(url, { waitUntil: 'networkidle', timeout: 60000 });
        await page.waitForTimeout(3000);
        
        const images = await page.evaluate(() => {
            const imgElements = document.querySelectorAll('img');
            const urls = [];
            imgElements.forEach((img, i) => {
                if (img.src && img.src.includes('substack')) {
                    urls.push({
                        index: i,
                        src: img.src,
                        alt: img.alt || ''
                    });
                }
            });
            return urls;
        });
        
        console.log(`Found ${images.length} images`);
        images.forEach(img => {
            console.log(`${img.index}: ${img.src.substring(0, 100)}...`);
        });
        
        return images;
    } catch (error) {
        console.error(`Error: ${error.message}`);
    } finally {
        await browser.close();
    }
}

(async () => {
    const blogs = [
        { name: 'Mamba', url: 'https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state' },
        { name: 'LLM Agents', url: 'https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents' },
        { name: 'Reasoning LLMs', url: 'https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms' },
        { name: 'Quantization', url: 'https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization' }
    ];
    
    for (const blog of blogs) {
        await extractImages(blog.url, blog.name);
    }
})();
