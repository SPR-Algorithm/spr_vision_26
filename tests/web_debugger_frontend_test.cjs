// Offline browser regression: mock both WebSocket endpoints, with no vehicle access.
// Requires Node.js and an existing Playwright >= 1.48 installation; no package install.
// Example (CHROME_BIN is optional when Playwright has its own Chromium):
// PLAYWRIGHT_MODULE_PATH=/path/to/playwright-core CHROME_BIN=/path/to/chrome \
//   node tests/web_debugger_frontend_test.cjs [index.html] [screenshot.png]
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE_PATH||'playwright-core');
const html=fs.readFileSync(path.resolve(process.argv[2]||path.join(__dirname,'../debug/static/index.html')),'utf8');
const screenshot=process.argv[3]?path.resolve(process.argv[3]):null;
(async()=>{
const browser=await chromium.launch({executablePath:process.env.CHROME_BIN,headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
try {
  const page=await browser.newPage({viewport:{width:1440,height:1100}});
  const errors=[]; page.on('pageerror',err=>errors.push(err.message));
  const videoConnections=[],telemetryConnections=[],videoCommands=[]; let videoClosed=0;
  await page.route('http://debugger.test/**',route=>route.fulfill({contentType:'text/html',body:html}));
  await page.routeWebSocket('ws://debugger.test/ws',ws=>{videoConnections.push(ws);ws.onMessage(message=>videoCommands.push(JSON.parse(message)));ws.onClose(()=>videoClosed++); ws.send(JSON.stringify({type:'stream',jpeg_quality:60,max_fps:30}));});
  await page.routeWebSocket('ws://debugger.test/ws/telemetry',ws=>{telemetryConnections.push(ws);ws.send(JSON.stringify({type:'telemetry_hello',capacity:256,batch_ms:50}));});
  await page.goto('http://debugger.test/');
  await page.waitForFunction(()=>telemetrySocket?.readyState===WebSocket.OPEN && ws?.readyState===WebSocket.OPEN);
  assert.equal(await page.locator('.chart-panel').count(),2);
  assert.equal(await page.locator('[data-field]').count(),30);
  assert.equal(await page.locator('[data-field]:checked').count(),6);

  const jpeg=Buffer.from((await page.evaluate(()=>{const c=document.createElement('canvas');c.width=320;c.height=180;const x=c.getContext('2d');x.fillStyle='#14314c';x.fillRect(0,0,320,180);return c.toDataURL('image/jpeg').split(',')[1];})),'base64');
  const metadata=Buffer.from(JSON.stringify({w:320,h:180,proc_ms:4.5,age_ms:10.5,dropped:0,jpeg_bytes:jpeg.length,detections:[],reprojections:[]}));
  const binaryFrame=Buffer.alloc(4+metadata.length+jpeg.length);binaryFrame.writeUInt32LE(metadata.length);metadata.copy(binaryFrame,4);jpeg.copy(binaryFrame,4+metadata.length);
  videoConnections[0].send(binaryFrame);
  await page.waitForFunction(()=>canvas.width===320&&canvas.height===180&&lastMeta?.w===320);
  assert.equal(await page.locator('#video-placeholder').isHidden(),true);
  videoConnections[0].send(JSON.stringify({type:'ekf_state',age_ms:1,data:{x:1,y:2,z:3,vx:.1,vy:.2,vz:.3,yaw:.1,w:.2,r:.3,l:.4,h:.5,v1:10,v2:20,R_yaw:.001,R_pitch:.002,R_distance:.03,nis:1,converged:true,active_param_set:'test',P_diag:[1,2],P_dim:2}}));
  await page.waitForFunction(()=>document.getElementById('ekf-pos').textContent.includes('1.000'));
  videoConnections[0].send(JSON.stringify({type:'param_sets',data:[{name:'test',active:true,v1:10,v2:20,R_yaw:.001,R_pitch:.002,R_distance:.03,angular_velocity_threshold:5}]}));
  await page.waitForSelector('#num-v1');
  await page.locator('.param-send-btn').click();
  await page.waitForTimeout(50);assert.equal(videoCommands.at(-1).cmd,'update_param');assert.equal(videoCommands.at(-1).v1,10);

  const makeSample=(seq)=>({seq,t:seq/100,monotonic_ms:100000000+seq*10,gimbal_yaw:Math.sin(seq/20),target_yaw:Math.sin(seq/20)+.04,plan_yaw:Math.sin(seq/20)-.02,gimbal_pitch:.3*Math.cos(seq/40),target_pitch:.3*Math.cos(seq/40)+.02,plan_pitch:.3*Math.cos(seq/40)-.01,control:true,target_valid:true,outpost_valid:false,fire:0,fired:0});
  telemetryConnections[0].send(JSON.stringify({type:'telemetry',samples:Array.from({length:3150},(_,i)=>makeSample(i+1)),dropped:{full:2,expired:3,client:4}}));
  await page.waitForFunction(()=>telemetryCount===3000);
  assert.deepEqual(await page.evaluate(()=>({first:telemetryAt(0).seq,last:telemetryAt(telemetryCount-1).seq,control:telemetryAt(0).control})),{first:151,last:3150,control:1});
  await page.waitForTimeout(120);
  assert.match(await page.locator('#telemetry-info').innerText(),/3000 \/ 3000/);
  if (screenshot) await page.screenshot({path:screenshot,fullPage:true});
  await page.waitForTimeout(550);
  assert.match(await page.locator('#telemetry-status').innerText(),/已过期/);
  await page.locator('#video-enabled').locator('..').locator('.slider').click();
  await page.waitForTimeout(2300);
  assert.equal(videoConnections.length,1); assert.equal(videoClosed,1);
  telemetryConnections[0].send(JSON.stringify({type:'telemetry',samples:[makeSample(3151)],dropped:{full:2,expired:3,client:4}}));
  await page.waitForFunction(()=>telemetryAt(telemetryCount-1).seq===3151);
  assert.match(await page.locator('#status').innerText(),/已暂停/);
  assert.match(await page.locator('#telemetry-status').innerText(),/实时/);
  await page.locator('#video-enabled').locator('..').locator('.slider').click();
  await page.waitForFunction(()=>ws?.readyState===WebSocket.OPEN); assert.equal(videoConnections.length,2);
  await page.locator('.field-picker summary').click();
  await page.locator('[data-field="control"]').check();
  assert.equal(await page.locator('.chart-panel').count(),3);
  await page.locator('#reset-fields').click(); assert.equal(await page.locator('.chart-panel').count(),2);
  const pathCheck=await page.evaluate(()=>{
   clearTelemetry();
   for (const sample of [
   {seq:1,t:1,monotonic_ms:1000,gimbal_yaw:1},
   {seq:2,t:1.01,monotonic_ms:1010,gimbal_yaw:null},
   {seq:4,t:1.03,monotonic_ms:1030,gimbal_yaw:2},
   {seq:5,t:1.04,monotonic_ms:1040,gimbal_yaw:3}]) appendTelemetry(sample);
   const panel=telemetryPanels[0],context=panel.canvas.getContext('2d');
   const move=context.moveTo.bind(context),line=context.lineTo.bind(context); const commands=[];
   context.strokeStyle='#000000';
   context.moveTo=(x,y)=>{if(context.strokeStyle==='#58a6ff')commands.push('move');return move(x,y);};
   context.lineTo=(x,y)=>{if(context.strokeStyle==='#58a6ff')commands.push('line');return line(x,y);};
   plotTelemetry(panel,0,1.04,10);
   context.moveTo=move;context.lineTo=line;
   return commands;
  });
  assert.deepEqual(pathCheck,['move','line','move','line','line']);
  await page.evaluate(()=>{
    const context=telemetryPanels[0].canvas.getContext('2d');
    window.frontendClear=context.clearRect.bind(context); window.frontendPaints=0;
    context.clearRect=(...args)=>{window.frontendPaints++;return window.frontendClear(...args);};
    Object.defineProperty(document,'hidden',{value:true,configurable:true}); telemetryDirty=true;
  });
  await page.waitForTimeout(150);
  assert.equal(await page.evaluate(()=>window.frontendPaints),0);
  await page.evaluate(()=>{delete document.hidden;document.dispatchEvent(new Event('visibilitychange'));});
  await page.waitForFunction(()=>window.frontendPaints>0);
  await page.evaluate(()=>{telemetryPanels[0].canvas.getContext('2d').clearRect=window.frontendClear;});
  telemetryConnections[0].close();
  await page.waitForFunction(()=>telemetrySocket?.readyState===WebSocket.OPEN);
  assert.equal(telemetryConnections.length,2);
  assert.equal(await page.evaluate(()=>telemetryCount),0);
  await page.setViewportSize({width:600,height:900});
  await page.waitForTimeout(100);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({ok:true,fields:30,ring:3000,defaults:6,checks:['independent pause/resume','telemetry stale','ring capacity','bool normalization','null+sequence line breaks','selection reset','hidden tab pauses chart painting','telemetry reconnect clears','mobile no horizontal overflow','binary JPEG video decode','EKF update','parameter command','no page errors'],screenshot}));
} finally {
  await browser.close();
}
})().catch(error=>{console.error(error);process.exitCode=1;});
