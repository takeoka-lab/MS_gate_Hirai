// Rebuild the atlas as editable PowerPoint objects in Artifact Session.
const W=1600,H=1131.4,FONT='Arial Unicode MS';
const ink='#192b42',muted='#526176';
function text(s,t,x,y,w,h,size=19,opts={}){
  const sh=s.shapes.add({geometry:'textbox',name:t.slice(0,80),position:{left:x,top:y,width:w,height:h,rotation:opts.rotation||0},fill:'none',line:{fill:'none',width:0}});
  sh.text=t;sh.text.style={typeface:FONT,fontSize:size,color:opts.color||ink,alignment:opts.align||'left',verticalAlignment:opts.valign||'top',insets:0,wrap:opts.wrap||'square',autoFit:'none',bold:opts.bold||false};return sh;
}
function shape(s,x,y,w,h,fill){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}});}
function footer(s,n){text(s,'MS GATE  /  2026-10-27',65,H-35,800,20,12,{color:muted});text(s,String(n),W-100,H-35,50,20,12,{align:'right',color:muted});}
function newSlide(p){return p.slides.add({width:W,height:H,background:{fill:'#ffffff'}});}
function chartFont(c){for(const st of [c.titleTextStyle,c.legend?.textStyle,c.dataLabels?.textStyle,c.xAxis?.textStyle,c.yAxis?.textStyle,c.xAxis?.title?.textStyle,c.yAxis?.title?.textStyle])if(st)st.typeface=FONT;}
function fmt(a){const span=a.ylim[1]-a.ylim[0],max=Math.max(...a.ylim.map(Math.abs));return a.yscale==='log'||(max>0&&max<.001)?'0.0E+00':span>=20?'0.##':span>=1?'0.###':span>=.01?'0.####':'0.000000';}
function niceAxis(lim,values){
  const span=lim[1]-lim[0],raw=span/5,pow=10**Math.floor(Math.log10(raw));
  const step=[1,2,2.5,5,10].find(v=>v*pow>=raw)*pow;
  let min=Math.floor(lim[0]/step)*step,max=Math.ceil(lim[1]/step)*step;
  if(values.every(v=>v>=0)&&min<0)min=0;
  if(values.every(v=>v<=0)&&max>0)max=0;
  return {min,max,majorUnit:step};
}
export function start(p,specs){
  p.slides.keep([]);
  let s=newSlide(p);text(s,'ノイズ相関・既存データ図集',66,62,1450,65,43,{bold:true});
  text(s,'初期平均フォノン数 / 運動位相緩和 / 運動加熱 / スピン位相緩和 / 光散乱',66,139,1450,40,24);
  const paragraphs=[
  '保存済みCSVから26図を再描画。新しいQPT計算は実行せず、既存の結果を背景条件別に整理した。各図の点が保存計算に対応する。線は視認性のために点を結んでいる。',
  '基準条件：単一モード矩形MSゲート、100 µs、eta=0.1、A=0.125、delta=0.5。主系列は n̄=0.01, 1, 2, 3, 4。レート倍率は0, 0.5, 1, 2, 4。use_full_order=True は有限LDのO(eta²)補正を含むモデルで、厳密な全次数ではない。',
  '基準レート：運動加熱10 /s、運動位相緩和18 /s、スピン位相緩和10/3 /s、光散乱4 /s（Rayleigh 3 + Raman 1）。全6ペアの格子は750点。共通軸を重複除去したQPTは565点。高温域の別検証はn̄=20まで保存されている。',
  '読み取り：温度増加に対して、無散逸の有限LD誤差と運動位相緩和の増分が大きく増える。他の3散逸の増分は比較的平坦。4散逸の同時作用はほぼ加算的で、最大の混合差分は運動加熱×運動位相緩和に現れる。',
  '相関の定義を区別する：(1) 温度と散逸の混合応答、(2) C_ij = I_ij − I_i − I_j + I_0 という非加算性、(3) 生成子内部のPauli間相関。C_ij < 0は単独増分の和より小さいことを表す。統計的に相関した環境浴を測定したものではない。',
  '生成子：K_0=log R_0、ΔK_i=log R_i−log R_0。保存された生成子解析はCPTP射影後。Hはコヒーレント項、Sは対角Pauli確率項、Cは対称なPauli交差項、Aは虚部の交差項。係数とノルムは誤差確率ではない。',
  '注意：微小差分のゼロレート外挿には不安定な条件がある。ペア生成子は上位3ペアの対角Pauliフィットまでで、全6ペアの完全H/S/C/A混合生成子や直接収束確認が揃ったことを意味しない。LD1の1e-16の下限値を数値精度として解釈しない。'];
  let y=220;for(const t of paragraphs){text(s,t,67,y,1460,100,20);y+=113;}
  s.speakerNotes.text='Source: output/pdf/noise_correlation_atlas_20261027.pdf, page 1. Data provenance: output/noise_correlation_20261027/manifest.json';footer(s,1);
  s=newSlide(p);text(s,'図の目次とデータの範囲',67,62,1460,60,36,{bold:true});
  for(const [i,a] of specs.entries()){text(s,`${String(i+1).padStart(2,'0')}  ${a.page.title}`,74,145+i*31,1370,27,17);text(s,String(i+3),1475,145+i*31,45,27,17,{align:'right'});}
  text(s,'元のCSV・設定JSONへのパスを各図とスピーカーノートに記載。入力のSHA-256と検算結果は元図集のmanifest.jsonを参照。',67,1004,1460,55,17);
  s.speakerNotes.text='Source: output/pdf/noise_correlation_atlas_20261027.pdf, page 2.';footer(s,2);
}

function chart(s,a,g,fig){
  const [x,y,w,h]=a.position;
  const px=g.x+x*g.w,py=g.y+(1-y-h)*g.h,pw=w*g.w,ph=h*g.h;
  // Native chart areas include titles, axis labels and legends around the plot.
  const compact=fig.axes.filter(v=>v.kind==='chart').length>=5;
  const left=px-(compact?62:77),top=py-35,width=pw+(compact?77:96),height=ph+(compact?84:96);
  const fs=compact?15:17,ss=compact?13:15;
  let title=a.title;
  if(a.yscale==='symlog')title+=' (linear scale)';
  const allX=a.series.flatMap(v=>v.x),allY=a.series.flatMap(v=>v.y);
  const xa=niceAxis(a.xlim,allX),ya=a.yscale==='log'?{min:a.ylim[0],max:a.ylim[1],logBase:10}:niceAxis(a.ylim,allY);
  if(Math.max(...allX)===4&&Math.min(...allX)>=0){xa.min=0;xa.max=4.1;xa.majorUnit=1;}
  const cfg={position:{left,top,width,height},title:title||undefined,titleTextStyle:{fontSize:fs+1,fill:ink},
    series:a.series.map(v=>({name:(a.legendTitle?a.legendTitle+' = ':'')+v.name,xValues:v.x,values:v.y,fill:v.color,line:{fill:v.scatter?'none':v.color,width:2.3,style:v.dash?'dashed':'solid'},marker:{symbol:v.marker?'circle':'none',size:5,fill:v.color}})),
    scatterOptions:{style:a.series.every(v=>v.scatter)?'marker':'lineWithMarkers'},hasLegend:a.legend,
    legend:{position:'bottom',overlay:false,textStyle:{fontSize:ss,fill:muted}},
    xAxis:{...xa,position:'bottom',tickLabelPosition:'low',crossesAt:xa.min,numberFormatCode:'0.##',title:{text:a.xlabel,textStyle:{fontSize:fs}},textStyle:{fontSize:ss,fill:muted},line:{fill:'#88939f',width:.8},majorGridlines:{fill:'#e5e9ee',width:.5}},
    yAxis:{...ya,crossesAt:ya.min,numberFormatCode:fmt(a),title:{text:a.ylabel,textStyle:{fontSize:fs}},textStyle:{fontSize:ss,fill:muted},line:{fill:'#88939f',width:.8},majorGridlines:{fill:'#e5e9ee',width:.5}},
    chartFill:'#ffffff',plotAreaFill:'#ffffff',chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}};
  const c=s.charts.add('scatter',cfg);chartFont(c);return c;
}

function heatmap(s,a,g){
  const [x,y,w,h]=a.position;const px=g.x+x*g.w,py=g.y+(1-y-h)*g.h,pw=w*g.w,ph=h*g.h;
  const n=a.values.length,m=a.values[0].length,cw=pw/m,ch=ph/n;
  const vals=a.values.slice().reverse(),colors=a.colors.slice().reverse();
  const t=s.tables.add({rows:n,columns:m,left:px,top:py,width:pw,height:ph,columnWidths:Array(m).fill(cw),values:vals.map(row=>row.map(v=>a.annotated?v.toFixed(2):''))});
  t.styleOptions={headerRow:false,totalRow:false,firstColumn:false,lastColumn:false,bandedRows:false,bandedColumns:false};
  t.borders.assign({fill:'none',width:0});
  t.cells.block({row:0,column:0,rowCount:n,columnCount:m}).assign({textStyle:{typeface:FONT,fontSize:a.annotated?11.8:9,alignment:'center'},margins:{left:0,right:0,top:0,bottom:0},anchor:'center'});
  for(let r=0;r<n;r++){t.rows[r].height=ch;for(let col=0;col<m;col++){const c=t.getCell(r,col),rgb=colors[r][col].match(/\w\w/g).map(v=>parseInt(v,16)),lum=rgb[0]*.2126+rgb[1]*.7152+rgb[2]*.0722;c.fill=colors[r][col];c.text.style={typeface:FONT,fontSize:a.annotated?11.8:9,alignment:'center',verticalAlignment:'middle',color:lum<140?'#ffffff':'#243246',insets:0,bold:false};}}
  if(a.title)text(s,a.title,px-10,py-37,pw+20,25,17,{align:'center'});
  a.xticklabels.forEach((v,i)=>text(s,v,px+i*cw-4,py+ph+8,cw+8,20,n>5?10:15,{align:'center'}));
  a.yticklabels.slice().reverse().forEach((v,i)=>text(s,v,px-40,py+i*ch+ch/2-10,34,20,n>5?10:15,{align:'right'}));
  if(a.xlabel)text(s,a.xlabel,px-10,py+ph+36,pw+20,25,15,{align:'center'});
  return t;
}
function colorbar(s,a,g){
  const [x,y,w,h]=a.position;const px=g.x+x*g.w,py=g.y+(1-y-h)*g.h,pw=w*g.w,ph=h*g.h;
  for(let i=0;i<40;i++){const ci=Math.round((1-i/39)*100);shape(s,px,py+i*ph/40,pw,ph/40+.2,a.colors[ci]);}
  for(const v of a.ticks){const yy=py+ph*(1-(v-a.vmin)/(a.vmax-a.vmin));text(s,Math.abs(v)>=1?v.toFixed(0):v.toPrecision(2),px+pw+6,yy-8,55,18,12);}
  const label=a.ylabel.replace(/\s+/g,' ');text(s,label,px-35,py-36,145,30,14,{align:'center'});
}

export function addFigure(p,fig){
  const s=newSlide(p),meta=fig.page;
  text(s,`${String(meta.number).padStart(2,'0')}  ${meta.title}`,70,43,1460,58,32,{bold:true});
  let subtitle=meta.subtitle.replace(/ \| /g,'  /  ');
  if(meta.number>=5&&meta.number<=10){const yy=fig.axes.find(a=>a.kind==='heatmap'&&a.ylabel)?.ylabel;subtitle+='  /  y: '+yy;}
  text(s,subtitle,70,105,1460,48,16,{color:muted});
  const ratio=fig.figsize[0]/fig.figsize[1],gw=Math.min(1504,826*ratio),gh=gw/ratio,g={x:(W-gw)/2,y:26,w:gw,h:gh};
  for(const a of fig.axes){
    if(a.kind==='chart')chart(s,a,g,fig);
    else if(a.kind==='heatmap')heatmap(s,a,g);
    else if(a.kind==='colorbar')colorbar(s,a,g);
    else if(a.kind==='text')for(const tt of a.texts){const [x,y,w,h]=a.position;text(s,tt.text,g.x+(x+tt.xy[0]*w)*g.w,g.y+(1-y-tt.xy[1]*h)*g.h,w*g.w,h*g.h,19);}
  }
  let cap=meta.caption;
  if(meta.number===21)cap+=' PowerPoint版の左上は線形軸。同じ元数値を使用。';
  const cy=Math.max(855,g.y+g.h+25);
  text(s,cap,65,cy,1470,135,18.5);
  const sourceY=Math.min(1046,cy+Math.ceil(cap.length/78)*25+12);
  text(s,meta.sources.map(x=>'出典: '+x).join('\n'),65,sourceY,1470,H-45-sourceY,12,{color:muted});
  s.speakerNotes.text=meta.caption+'\n\nSource PDF: output/pdf/noise_correlation_atlas_20261027.pdf, page '+(meta.number+2)+'\n'+meta.sources.join('\n')+'\n\n'+(meta.number===21?'The stochastic-shape panel uses a linear y-axis in PowerPoint; all raw values are retained.':'')+fig.axes.filter(a=>a.kind==='heatmap').map((a,i)=>'\nHeatmap '+(i+1)+' raw values, bottom row first:\n'+a.values.map(row=>row.join(',')).join('\n')).join('');
  footer(s,meta.number+2);return s;
}
export const dimensions={width:W,height:H,font:FONT};
