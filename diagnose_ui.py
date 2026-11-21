import asyncio
import json
import os
import subprocess
import time
import urllib.request
import websockets
import base64

SCREENSHOT_DIR = r"D:\ASTRA_SCOUT\screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
USER_DATA = os.path.join(os.environ.get("TEMP", "C:\\Temp"), f"chrome_cdp_{int(time.time())}")

async def send_cdp(ws, method, params=None, msg_id=1):
    payload = {"id": msg_id, "method": method, "params": params or {}}
    await ws.send(json.dumps(payload))
    while True:
        resp = await ws.recv()
        data = json.loads(resp)
        if data.get("id") == msg_id:
            return data

async def wait_until_idle(ws, timeout=25):
    """Waits until Streamlit finishes running and content is rendered."""
    js_check = """
    (() => {
        const stopBtn = document.querySelector('button[kind="header"]');
        const running = stopBtn && stopBtn.innerText.includes('Stop');
        const metrics = document.querySelectorAll('[data-testid="stMetric"]');
        const h1 = document.querySelector('h1');
        return {
            ready: !running && (h1 !== null),
            metricCount: metrics.length,
            title: h1 ? h1.innerText : ''
        };
    })()
    """
    start_t = time.time()
    while time.time() - start_t < timeout:
        res = await send_cdp(ws, "Runtime.evaluate", {"expression": js_check, "returnByValue": True}, msg_id=999)
        val = res.get("result", {}).get("result", {}).get("value", {})
        if val.get("ready"):
            return val
        await asyncio.sleep(1.0)
    return {}

async def run_diagnostics():
    chrome_proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        "--remote-debugging-port=9222",
        f"--user-data-dir={USER_DATA}",
        "--disable-gpu",
        "--no-sandbox",
        "--window-size=1366,900"
    ])
    print("Launched Chrome with CDP on port 9222...")
    time.sleep(2)

    try:
        req = urllib.request.urlopen("http://127.0.0.1:9222/json")
        targets = json.loads(req.read().decode())
        ws_url = targets[0]["webSocketDebuggerUrl"]
        print(f"Connected to CDP target: {ws_url}")

        pages = [
            ("home", "app"),
            ("mission_control", "Mission Control"),
            ("terrain_analysis", "Terrain Analysis"),
            ("scouting_intelligence", "Scouting Intelligence"),
            ("route_comparison", "Route Comparison"),
            ("about_project", "About Project")
        ]

        async with websockets.connect(ws_url, max_size=30*1024*1024) as ws:
            await send_cdp(ws, "Page.enable", msg_id=1)
            await send_cdp(ws, "DOM.enable", msg_id=2)
            await send_cdp(ws, "Runtime.enable", msg_id=3)

            # 1. Initial navigation to app and wait for simulation initialization
            print("Loading home page (initializing simulation session state)...")
            await send_cdp(ws, "Page.navigate", {"url": "http://localhost:8501/"}, msg_id=10)
            status = await wait_until_idle(ws, timeout=25)
            print(f"Home page ready: {status}")

            for width in [1366, 900]:
                print(f"\n==================== VIEWPORT WIDTH: {width}px ====================")
                await send_cdp(ws, "Emulation.setDeviceMetricsOverride", {
                    "width": width,
                    "height": 900,
                    "deviceScaleFactor": 1,
                    "mobile": False
                }, msg_id=20)
                await asyncio.sleep(1.0)

                for name, link_text in pages:
                    print(f"\nNavigating to {name} by clicking sidebar link '{link_text}'...")
                    js_click = f"""
                    (() => {{
                        const links = Array.from(document.querySelectorAll('a[data-testid="stSidebarNavLink"]'));
                        const link = links.find(l => l.innerText.toLowerCase().includes('{link_text.lower()}'));
                        if (link) {{
                            link.click();
                            return true;
                        }}
                        return false;
                    }})()
                    """
                    click_res = await send_cdp(ws, "Runtime.evaluate", {"expression": js_click, "returnByValue": True}, msg_id=30)
                    clicked = click_res.get("result", {}).get("result", {}).get("value")
                    if not clicked:
                        # Direct URL fallback
                        url_suffix = "" if name == "home" else name.title()
                        await send_cdp(ws, "Page.navigate", {"url": f"http://localhost:8501/{url_suffix}"}, msg_id=31)

                    await wait_until_idle(ws, timeout=20)
                    await asyncio.sleep(2.5)  # Render buffer

                    # Capture screenshot
                    shot_res = await send_cdp(ws, "Page.captureScreenshot", {"format": "png"}, msg_id=40)
                    img_data = shot_res["result"]["data"]
                    shot_path = os.path.join(SCREENSHOT_DIR, f"{name}_{width}_after.png")
                    with open(shot_path, "wb") as f:
                        f.write(base64.b64decode(img_data))
                    print(f"Saved screenshot: {shot_path} ({os.path.getsize(shot_path)} bytes)")

                    # Detailed computed CSS & layout inspection
                    js_inspect = """
                    (() => {
                        const items = [];
                        const overlaps = [];

                        const selectors = [
                            'h1', 'h2', 'h3', 'h4',
                            '[data-testid="stCaptionContainer"]',
                            '[data-testid="stMetric"]',
                            '[data-testid="stMetricLabel"]',
                            '[data-testid="stMetricValue"]',
                            '.hero-subtitle'
                        ];

                        selectors.forEach(sel => {
                            document.querySelectorAll(sel).forEach(el => {
                                const r = el.getBoundingClientRect();
                                const s = window.getComputedStyle(el);
                                if (r.width > 0 && r.height > 0) {
                                    items.push({
                                        tag: el.tagName,
                                        sel: sel,
                                        text: (el.innerText || '').substring(0, 35).replace(/\\n/g, ' '),
                                        top: Math.round(r.top),
                                        bottom: Math.round(r.bottom),
                                        left: Math.round(r.left),
                                        right: Math.round(r.right),
                                        width: Math.round(r.width),
                                        height: Math.round(r.height),
                                        fontSize: s.fontSize,
                                        lineHeight: s.lineHeight,
                                        marginTop: s.marginTop,
                                        marginBottom: s.marginBottom,
                                        padding: s.padding,
                                        position: s.position,
                                        el: el
                                    });
                                }
                            });
                        });

                        // Check vertical overlaps
                        for (let i = 0; i < items.length; i++) {
                            for (let j = i + 1; j < items.length; j++) {
                                const a = items[i];
                                const b = items[j];
                                if (a.el.contains(b.el) || b.el.contains(a.el)) continue;

                                const xOverlap = (a.left < b.right && a.right > b.left);
                                if (xOverlap) {
                                    if (a.top <= b.top && a.bottom > b.top) {
                                        overlaps.push({
                                            a: `${a.tag} [${a.text}] (${a.top}-${a.bottom})`,
                                            b: `${b.tag} [${b.text}] (${b.top}-${b.bottom})`,
                                            overlapPx: a.bottom - b.top,
                                            aCSS: `font: ${a.fontSize}, lh: ${a.lineHeight}, m-bot: ${a.marginBottom}`,
                                            bCSS: `font: ${b.fontSize}, lh: ${b.lineHeight}, m-top: ${b.marginTop}`
                                        });
                                    } else if (b.top <= a.top && b.bottom > a.top) {
                                        overlaps.push({
                                            a: `${b.tag} [${b.text}] (${b.top}-${b.bottom})`,
                                            b: `${a.tag} [${a.text}] (${a.top}-${a.bottom})`,
                                            overlapPx: b.bottom - a.top,
                                            aCSS: `font: ${b.fontSize}, lh: ${b.lineHeight}, m-bot: ${b.marginBottom}`,
                                            bCSS: `font: ${a.fontSize}, lh: ${a.lineHeight}, m-top: ${a.marginTop}`
                                        });
                                    }
                                }
                            }
                        }

                        return {
                            count: items.length,
                            overlaps: overlaps,
                            items: items.map(n => ({
                                tag: n.tag,
                                sel: n.sel,
                                text: n.text,
                                top: n.top,
                                bottom: n.bottom,
                                fontSize: n.fontSize,
                                lineHeight: n.lineHeight,
                                marginTop: n.marginTop,
                                marginBottom: n.marginBottom,
                                position: n.position
                            }))
                        };
                    })()
                    """
                    inspect_res = await send_cdp(ws, "Runtime.evaluate", {"expression": js_inspect, "returnByValue": True}, msg_id=50)
                    val = inspect_res["result"]["result"]["value"]
                    print(f"Total elements inspected on {name}: {val['count']}")
                    if val["overlaps"]:
                        print(f"⚠️ FOUND {len(val['overlaps'])} OVERLAPS:")
                        for ov in val["overlaps"]:
                            print(f"  - Overlap ({ov['overlapPx']}px): {ov['a']} vs {ov['b']}")
                            print(f"    Top CSS: {ov['aCSS']}")
                            print(f"    Bottom CSS: {ov['bCSS']}")
                    else:
                        print(f"✅ Clean: 0 overlaps detected on {name}.")

                    for itm in val["items"][:4]:
                        print(f"    [{itm['tag']}] '{itm['text']}' -> top:{itm['top']}, bot:{itm['bottom']}, font:{itm['fontSize']}, lh:{itm['lineHeight']}, m-top:{itm['marginTop']}, m-bot:{itm['marginBottom']}")

    finally:
        chrome_proc.terminate()
        print("Terminated Chrome CDP process.")

if __name__ == "__main__":
    asyncio.run(run_diagnostics())
