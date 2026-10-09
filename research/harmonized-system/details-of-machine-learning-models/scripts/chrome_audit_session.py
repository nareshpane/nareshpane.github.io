"""Small standard-library Chrome DevTools client for local-page QA only."""
import base64, hashlib, json, os, socket, struct, subprocess, time
from urllib.request import urlopen

class ChromeSession:
    def __init__(self,browser,profile,width=1440):
        self.process=subprocess.Popen([str(browser),'--headless','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-port=0',f'--user-data-dir={profile}',f'--window-size={width},1100','--force-device-scale-factor=1','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        try:self.connect(profile)
        except Exception:
            self.process.terminate();self.process.wait(timeout=10)
            raise

    def connect(self,profile):
        ready=profile/'DevToolsActivePort'
        port=None
        for _ in range(100):
            try:
                if ready.exists():port=int(ready.read_text().splitlines()[0]);break
            except (PermissionError,IndexError,ValueError):pass
            time.sleep(.1)
        assert port,'Chrome did not expose its local debugging port'
        targets=json.loads(urlopen(f'http://localhost:{port}/json').read())
        url=next(x['webSocketDebuggerUrl'] for x in targets if x['type']=='page')
        from urllib.parse import urlparse
        u=urlparse(url);self.sock=socket.create_connection((u.hostname,u.port),timeout=30)
        key=base64.b64encode(os.urandom(16)).decode()
        request=f'GET {u.path} HTTP/1.1\r\nHost: {u.netloc}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n'
        self.sock.sendall(request.encode());self.buffer=b''
        while b'\r\n\r\n' not in self.buffer:self.buffer+=self.sock.recv(4096)
        header,self.buffer=self.buffer.split(b'\r\n\r\n',1)
        expected=base64.b64encode(hashlib.sha1((key+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest())
        assert b' 101 ' in header and expected in header,'WebSocket handshake failed: '+header.decode(errors='replace')
        self.sequence=0

    def read(self,n):
        while len(self.buffer)<n:
            part=self.sock.recv(max(4096,n-len(self.buffer)))
            if not part:raise RuntimeError('Chrome connection closed')
            self.buffer+=part
        result,self.buffer=self.buffer[:n],self.buffer[n:]
        return result

    def send(self,payload,opcode=1):
        mask=os.urandom(4);n=len(payload)
        length=bytes([n|128]) if n<126 else bytes([126|128])+struct.pack('!H',n) if n<65536 else bytes([127|128])+struct.pack('!Q',n)
        self.sock.sendall(bytes([128|opcode])+length+mask+bytes(x^mask[i%4] for i,x in enumerate(payload)))

    def message(self):
        chunks=[]
        while True:
            a,b=self.read(2);n=b&127
            if n==126:n=struct.unpack('!H',self.read(2))[0]
            elif n==127:n=struct.unpack('!Q',self.read(8))[0]
            mask=self.read(4) if b&128 else None
            payload=self.read(n)
            if mask:payload=bytes(x^mask[i%4] for i,x in enumerate(payload))
            if a&15==9:self.send(payload,10);continue
            if a&15==8:raise RuntimeError('Chrome WebSocket closed')
            chunks.append(payload)
            if a&128:return json.loads(b''.join(chunks))

    def call(self,method,params=None):
        self.sequence+=1;request={'id':self.sequence,'method':method,'params':params or {}}
        self.send(json.dumps(request).encode())
        while True:
            reply=self.message()
            if reply.get('id')!=self.sequence:continue
            if 'error' in reply:raise RuntimeError(reply['error'])
            return reply.get('result',{})

    def capture(self,url,path,width):
        self.call('Emulation.setDeviceMetricsOverride',{'width':max(width,500),'height':1100,'deviceScaleFactor':1,'mobile':False})
        self.call('Page.navigate',{'url':url})
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            expression='location.href === '+json.dumps(url)+" ? (document.getElementById('qa')?.textContent || '') : ''"
            result=self.call('Runtime.evaluate',{'expression':expression,'returnByValue':True})
            value=result.get('result',{}).get('value','')
            if value:
                report=json.loads(value)
                # Let the fragment layout and its MathJax SVG finish painting.
                self.call('Runtime.evaluate',{'expression':'new Promise(resolve=>setTimeout(resolve,100))','awaitPromise':True})
                shot=self.call('Page.captureScreenshot',{'format':'png','captureBeyondViewport':False})
                path.write_bytes(base64.b64decode(shot['data']))
                return report
            time.sleep(.1)
        raise RuntimeError('The local browser audit did not finish within 60 seconds')

    def close(self):
        self.sock.close()
        self.process.terminate()
        self.process.wait(timeout=10)
