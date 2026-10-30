"""Assemble the graph atlas with Japanese captions; no scientific computation."""
from pathlib import Path
import json
import html
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.colors import HexColor
from PIL import Image

OUT=Path(__file__).resolve().parent
DEST=OUT.parent/"pdf"/"noise_correlation_atlas_20261027.pdf"
DEST.parent.mkdir(exist_ok=True)
pages=json.loads((OUT/"figures.json").read_text())
manifest=json.loads((OUT/"manifest.json").read_text())
pdfmetrics.registerFont(TTFont("JP","/System/Library/Fonts/Supplemental/Arial Unicode.ttf"))
W,H=1190.55,841.89  # A3 landscape: legible 5-column grid cells.
c=canvas.Canvas(str(DEST),pagesize=(W,H))
c.setTitle("ノイズ相関・既存データ図集 2026-10-27")
c.setAuthor("MS_gate_Hirai / saved-data analysis")
style=ParagraphStyle("jp",fontName="JP",fontSize=14,leading=23,textColor=HexColor("#25334b"),wordWrap="CJK")
small=ParagraphStyle("small",parent=style,fontSize=10,leading=15)

def text(txt,x,y,width=W-100,st=style):
    p=Paragraph(txt,st);_,height=p.wrap(width,H);p.drawOn(c,x,y-height);return y-height

def footer(page):
    c.setStrokeColor(HexColor("#dde3e9"));c.line(48,38,W-48,38)
    c.setFillColor(HexColor("#64748b"));c.setFont("JP",10)
    c.drawString(48,22,"MS GATE | SAVED-DATA ATLAS | 2026-10-27")
    c.drawRightString(W-48,22,str(page))

c.setFillColor(HexColor("#192b42"));c.setFont("JP",32);c.drawString(50,H-78,"ノイズ相関・既存データ図集")
c.setFont("JP",18);c.drawString(50,H-115,"初期平均フォノン数 / 運動位相緩和 / 運動加熱 / スピン位相緩和 / 光散乱")
y=H-155
intro=[
f"保存済みCSVから{len(pages)}図を再描画。新しいQPT計算は実行せず、既存の結果を背景条件別に整理した。各図の点が保存計算に対応する。線は視認性のために点を結んでいる。",
"基準条件：単一モード矩形MSゲート、100 µs、eta=0.1、A=0.125、delta=0.5。主系列は nbar=0.01, 1, 2, 3, 4。レート倍率は0, 0.5, 1, 2, 4。use_full_order=True は有限LDのO(eta²)補正を含むモデルで、厳密な全次数ではない。",
"基準レート：運動加熱10 /s、運動位相緩和18 /s、スピン位相緩和10/3 /s、光散乱4 /s（Rayleigh 3 + Raman 1）。全6ペアの格子は750点。共通軸を重複除去したQPTは565点。高温域の別検証はnbar=20まで保存されている。",
"読み取り：温度増加に対して、無散逸の有限LD誤差と運動位相緩和の増分が大きく増える。他の3散逸の増分は比較的平坦。4散逸の同時作用はほぼ加算的で、最大の混合差分は運動加熱×運動位相緩和に現れる。",
"相関の定義を区別する：(1) 温度と散逸の混合応答、(2) C_ij = I_ij - I_i - I_j + I_0 という非加算性、(3) 生成子内部のPauli間相関。C_ij&lt;0は単独増分の和より小さいことを表す。統計的に相関した環境浴を測定したものではない。",
"生成子：K_0=log R_0、ΔK_i=log R_i-log R_0。保存された生成子解析はCPTP射影後。Hはコヒーレント項、Sは対角Pauli確率項、Cは対称なPauli交差項、Aは虚部の交差項。係数とノルムは誤差確率ではない。",
"注意：微小差分のゼロレート外挿には不安定な条件がある。ペア生成子は上位3ペアの対角Pauliフィットまでで、全6ペアの完全H/S/C/A混合生成子や直接収束確認が揃ったことを意味しない。LD1の1e-16の下限値を数値精度として解釈しない。",
]
for t in intro: y=text(t,50,y)-18
footer(1);c.showPage()
c.setFont("JP",26);c.setFillColor(HexColor("#192b42"));c.drawString(50,H-65,"図の目次とデータの範囲")
y=H-105
for page in pages:
    t=f'{page["number"]:02d}  {page["title"]}'
    c.setFont("JP",12);c.drawString(55,y,t);c.drawRightString(W-55,y,str(page["number"]+2));y-=23
text("元のCSV・設定JSONへのパスを各図末尾に記載。全入力のSHA-256と検算結果はmanifest.json、主要数値はdata/、PNGとSVGはfigures/に保存。スクリプトは元結果を変更せず再実行可能。",50,115,st=small)
footer(2);c.showPage()
for page in pages:
    path=OUT/page["png"]
    iw,ih=Image.open(path).size
    avail_w,avail_h=W-64,620
    scale=min(avail_w/iw,avail_h/ih);dw,dh=iw*scale,ih*scale
    c.drawImage(str(path),(W-dw)/2,H-20-dh,width=dw,height=dh)
    y=H-30-dh
    y=text(page["caption"],48,y,W-96)-12
    for p in page["sources"]:
        y=text("出典: "+html.escape(p),48,y,W-96,small)-3
    if y<45: raise RuntimeError(f'Caption overflow: {page["number"]}, y={y}')
    footer(page["number"]+2);c.showPage()
c.save()
print(str(DEST))
