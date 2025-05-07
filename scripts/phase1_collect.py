#!/usr/bin/env python3
import os
import re
import json
import gzip
import pickle
from tqdm import tqdm
from bs4 import BeautifulSoup

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
BASE_DIR        = "/Users/akshat/Data/UIUC/Spring 2025/Courses/CS 568 User-Centered Machine Learning/Project/WebUI-7k/train_split_web7k"
VIEWPORTS       = ["1280-720","1366-768","1536-864","1920-1080","iPad-Pro","iPhone-13 Pro"]
os.makedirs("intermediate", exist_ok=True)  # folder for intermediate pickle

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def load_json(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.load(open(path, encoding="utf-8"))

def parse_rgba(s):
    if not s: return None
    s = s.strip().lower()
    if s=="transparent": return None
    m = re.match(r"rgba\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)", s)
    if m:
        r,g,b,a = m.groups()
        if float(a)==0: return None
        return (int(r),int(g),int(b))
    m = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", s)
    return tuple(int(x) for x in m.groups()) if m else None

def contrast_ratio(fg,bg):
    def lum(c):
        v=c/255.0
        return v/12.92 if v<=0.03928 else ((v+0.055)/1.055)**2.4
    L1=0.2126*lum(fg[0])+0.7152*lum(fg[1])+0.0722*lum(fg[2])
    L2=0.2126*lum(bg[0])+0.7152*lum(bg[1])+0.0722*lum(bg[2])
    l,d=max(L1,L2),min(L1,L2)
    return (l+0.05)/(d+0.05)

# ─── PHASE 1 ─────────────────────────────────────────────────────────────────────
def main():
    per_page = {}
    axe_jobs = []

    for pid in tqdm(os.listdir(BASE_DIR), desc="Pages"):
        page_dir = os.path.join(BASE_DIR, pid)
        if not os.path.isdir(page_dir): continue

        result = {'page_id': pid, 'viewports': []}
        for vp in VIEWPORTS:
            prefix = vp if vp in ("iPad-Pro","iPhone-13 Pro") else f"default_{vp}"
            needed = [f"{prefix}-html.html",
                      f"{prefix}-axtree.json.gz",
                      f"{prefix}-viewport.json.gz",
                      f"{prefix}-style.json.gz",
                      f"{prefix}-bb.json.gz"]
            if any(not os.path.exists(os.path.join(page_dir,n)) for n in needed):
                continue

            HTML = os.path.join(page_dir, f"{prefix}-html.html")
            AXT  = load_json(os.path.join(page_dir, f"{prefix}-axtree.json.gz"))['nodes']
            VP   = load_json(os.path.join(page_dir, f"{prefix}-viewport.json.gz"))
            STY  = load_json(os.path.join(page_dir, f"{prefix}-style.json.gz"))
            BB   = load_json(os.path.join(page_dir, f"{prefix}-bb.json.gz"))

            # screenshot?
            ss = ""
            for nm in (f"{prefix}-screenshot-full.webp", f"{prefix}-screenshot.webp"):
                p = os.path.join(page_dir, nm)
                if os.path.exists(p):
                    ss = p; break

            by_back = {n['backendDOMNodeId']:n for n in AXT if n.get('backendDOMNodeId')!=None}
            by_id   = {n['nodeId']:n for n in AXT}

            def fg(bid): return parse_rgba(STY.get(str(bid),{}).get('color'))
            def bg(bid):
                c = parse_rgba(STY.get(str(bid),{}).get('background-color'))
                if c: return c
                node = by_back.get(bid)
                while node:
                    node = by_id.get(node.get('parentId'))
                    if not node: break
                    c2 = parse_rgba(STY.get(str(node['backendDOMNodeId']),{}).get('background-color'))
                    if c2: return c2
                return (255,255,255)

            # 1) semantic
            try:
                soup = BeautifulSoup(open(HTML,encoding='utf-8'), "html.parser")
                lang = (soup.html or {}).get('lang','')
                headings = [[int(h.name[1]), h.get_text(strip=True)]
                            for h in soup.find_all(re.compile(r"^h[1-6]$"))]
                images=[]; missing_alt=[]
                for bid,node in by_back.items():
                    if node['role']['value']=='img':
                        alt = node.get('name',{}).get('value','')
                        images.append({'nodeId':str(bid),'alt':alt})
                        if not alt.strip(): missing_alt.append(str(bid))
                images.sort(key=lambda x:int(x['nodeId']))
                links=[]; missing_name=[]
                for bid,node in by_back.items():
                    if node['role']['value']=='link':
                        txt=node.get('name',{}).get('value','')
                        links.append({'nodeId':str(bid),'text':txt})
                        if not txt.strip(): missing_name.append(str(bid))
                links.sort(key=lambda x:int(x['nodeId']))
                semantic = {'lang':lang,
                            'headings':headings,
                            'images':images,'missing_alt':missing_alt,
                            'links':links,'missing_name':missing_name}
            except:
                semantic={'lang':'','headings':[],'images':[],'missing_alt':[],
                          'links':[],'missing_name':[]}

            # 2) contrast
            try:
                TEXT_ROLES={'staticText','link','heading','text'}
                contrast=[]
                for node in AXT:
                    bid=node.get('backendDOMNodeId')
                    if not bid or not VP.get(str(bid),False): continue
                    r=node['role']['value']
                    if r not in TEXT_ROLES: continue
                    fgc=fg(bid)
                    if not fgc: continue
                    contrast.append({
                      'role':r,'backendId':bid,
                      'fg':fgc,'bg':bg(bid),
                      'contrast':contrast_ratio(fgc,bg(bid))
                    })
            except:
                contrast=[]

            # 3) image-captioning
            try:
                image_captioning=[]
                for img in semantic['images']:
                    bbx=BB.get(img['nodeId'])
                    if bbx:
                        image_captioning.append({
                          'nodeId':img['nodeId'],
                          'alt':img['alt'],
                          'bbox':bbx
                        })
            except:
                image_captioning=[]

            # project entry
            vp_entry = {
              'viewport':vp,
              'semantic':semantic,
              'contrast':contrast,
              'image_captioning':image_captioning,
              'axe':None,
              'html_path':HTML,
              'screenshot':ss
            }
            result['viewports'].append(vp_entry)

            # schedule axe
            axe_jobs.append({
              'htmlUrl':'file://'+HTML,
              'outFile':os.path.join(page_dir,f"{prefix}-axe.json"),
              'pageId':pid,
              'vpIndex':len(result['viewports'])-1
            })

        per_page[pid]=result

    # write out
    json.dump(axe_jobs, open("axe_jobs.json","w"), indent=2)
    pickle.dump(per_page, open("intermediate/per_page.pkl","wb"))
    print("Phase 1 done: wrote axe_jobs.json + intermediate/per_page.pkl")

if __name__=="__main__":
    main()