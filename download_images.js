const { chromium } = require('playwright');
const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');
const { URL } = require('url');

async function downloadImage(url, filepath) {
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(filepath);
        const protocol = url.startsWith('https') ? https : http;
        
        protocol.get(url, (response) => {
            if (response.statusCode === 301 || response.statusCode === 302) {
                downloadImage(response.headers.location, filepath).then(resolve).catch(reject);
                return;
            }
            response.pipe(file);
            file.on('finish', () => {
                file.close();
                resolve();
            });
        }).on('error', (err) => {
            fs.unlink(filepath, () => {});
            reject(err);
        });
    });
}

function cleanUrl(url) {
    let cleaned = url.replace(/,w_\d+,c_limit/g, '');
    cleaned = cleaned.replace(/,w_\d+,h_\d+,c_fill/g, '');
    cleaned = cleaned.replace(/,fl_lossy/g, ',fl_progressive:steep');
    return cleaned;
}

async function extractAndDownloadImages(url, blogName, outputDir, startIndex, endIndex) {
    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    
    console.log(`\n=== ${blogName} ===`);
    
    try {
        await page.goto(url, { waitUntil: 'networkidle', timeout: 60000 });
        await page.waitForTimeout(3000);
        
        const images = await page.evaluate(() => {
            const imgElements = document.querySelectorAll('img');
            const urls = [];
            imgElements.forEach((img, i) => {
                if (img.src && img.src.includes('substackcdn.com/image/fetch') && img.src.includes('w_1456,c_limit')) {
                    urls.push({
                        index: i,
                        src: img.src
                    });
                }
            });
            return urls;
        });
        
        console.log(`Found ${images.length} main images to download`);
        
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }
        
        let successCount = 0;
        let failCount = 0;
        
        for (let i = 0; i < images.length; i++) {
            const img = images[i];
            const cleanedUrl = cleanUrl(img.src);
            const ext = cleanedUrl.includes('.png') ? 'png' : cleanedUrl.includes('.jpg') || cleanedUrl.includes('.jpeg') ? 'jpg' : 'gif';
            const filename = `${blogName.toLowerCase().replace(/ /g, '-')}_${String(i + 1).padStart(3, '0')}.${ext}`;
            const filepath = path.join(outputDir, filename);
            
            try {
                await downloadImage(cleanedUrl, filepath);
                console.log(`  [${i + 1}/${images.length}] Downloaded: ${filename}`);
                successCount++;
            } catch (err) {
                console.error(`  [${i + 1}/${images.length}] Failed: ${filename} - ${err.message}`);
                failCount++;
            }
            
            await new Promise(r => setTimeout(r, 500));
        }
        
        console.log(`\n${blogName}: ${successCount} succeeded, ${failCount} failed`);
        return { success: successCount, failed: failCount };
        
    } catch (error) {
        console.error(`Error: ${error.message}`);
    } finally {
        await browser.close();
    }
}

(async () => {
    const blogs = [
        { 
            name: 'Mamba', 
            url: 'https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state',
            dir: 'assets/mamba'
        },
        { 
            name: 'LLM Agents', 
            url: 'https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents',
            dir: 'assets/llm-agents'
        },
        { 
            name: 'Reasoning LLMs', 
            url: 'https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms',
            dir: 'assets/reasoning-llms'
        },
        { 
            name: 'Quantization', 
            url: 'https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization',
            dir: 'assets/quantization'
        }
    ];
    
    const results = [];
    for (const blog of blogs) {
        const result = await extractAndDownloadImages(blog.url, blog.name, blog.dir);
        results.push({ blog: blog.name, ...result });
    }
    
    console.log('\n\n=== SUMMARY ===');
    results.forEach(r => {
        console.log(`${r.blog}: ${r.success} succeeded, ${r.failed} failed`);
    });
})();
