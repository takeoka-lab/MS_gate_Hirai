import fs from 'node:fs/promises';
import path from 'node:path';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
import {start,addFigure,dimensions} from './build_editable_deck.mjs';
const build=path.dirname(new URL(import.meta.url).pathname);
const dir=decodeURIComponent(build);
const specs=JSON.parse(await fs.readFile(path.join(dir,'plot_specs.json'),'utf8'));
const p=Presentation.create({slideSize:{width:dimensions.width,height:dimensions.height}});
start(p,specs);
for(const fig of specs){addFigure(p,fig);console.log(`Built figure ${fig.page.number}`);}
await fs.mkdir(path.join(dir,'previews'),{recursive:true});
await (await PresentationFile.exportPptx(p)).save(path.join(dir,'candidate.pptx'));
console.log('Exported candidate.pptx');
for(let i=0;i<p.slides.items.length;i++){
  const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
  await fs.writeFile(path.join(dir,'previews',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
  console.log(`Rendered slide ${i+1}`);
}
await fs.writeFile(path.join(dir,'expected_counts.json'),JSON.stringify({slides:28,charts:72,tables:65,width:dimensions.width,height:dimensions.height}));
