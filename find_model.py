import urllib.request
import re

url = 'https://html.duckduckgo.com/html/?q=mobile_face_net.tflite+site:raw.githubusercontent.com'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
try:
    resp = urllib.request.urlopen(req)
    html = resp.read().decode('utf-8')
    links = re.findall(r'href="(https://[^"]*mobile_face_net\.tflite[^"]*)"', html)
    if not links:
        # try without mobile_face_net.tflite
        url2 = 'https://html.duckduckgo.com/html/?q=mobilefacenet.tflite+site:raw.githubusercontent.com'
        req2 = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        html2 = urllib.request.urlopen(req2).read().decode('utf-8')
        links.extend(re.findall(r'href="(https://[^"]*mobilefacenet\.tflite[^"]*)"', html2))
    
    print("Found links:")
    for link in links:
        print(link)
except Exception as e:
    print(e)

