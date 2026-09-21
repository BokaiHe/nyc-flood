import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runtime = process.env.RUNTIME_NODE_MODULES;
const { Presentation, PresentationFile } = await import(pathToFileURL(path.join(runtime,'@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const out = path.join(root,'outputs/method_figures');
const assets = path.join(out,'assets');
const ppt = Presentation.create({slideSize:{width:1600,height:700}});
const FONT='Arial';
const C={ink:'#111111',line:'#333333',blue:'#3775BA',red:'#B64342',rose:'#F8DCD9',ice:'#D9E8F8',green:'#E7F1E9'};
let count=0;
function shape(slide,x,y,w,h,{fill='#FFFFFF',line=C.line,geometry='roundRect'}={}){
 return slide.shapes.add({name:`element-${++count}`,geometry,position:{left:x,top:y,width:w,height:h},fill,
 line:{fill:line,width:1.0},...(geometry==='roundRect'?{borderRadius:12}:{})});
}
function txt(s,x,y,w,h,t,size=22,bold=false,color=C.ink,align='center'){
 const sh=shape(s,x,y,w,h,{fill:'none',line:'none',geometry:'textbox'});sh.text=t;
 sh.text.style={typeface:FONT,fontSize:size,bold,color,alignment:align,verticalAlignment:'middle',autoFit:'none',insets:{top:0,right:0,bottom:0,left:0}};return sh;
}
function box(s,x,y,w,h,title){const b=shape(s,x,y,w,h);txt(s,x+10,y+9,w-20,35,title,24,true);return b;}
async function img(s,name,x,y,size){s.images.add({blob:new Uint8Array(await fs.readFile(path.join(assets,name))),contentType:'image/png',alt:name,fit:'contain',position:{left:x,top:y,width:size,height:size}});}
function link(s,a,b,{from='right',to='left',dash=false,kind='straight',both=false,arrow=true}={}){
 const connector=s.shapes.connect(a,b,{fromSide:from,toSide:to,kind,line:{fill:C.line,width:1.3,style:dash?'dashed':'solid'},...(arrow?{tail:{type:'triangle',width:'med',length:'med'}}:{}),...(both?{head:{type:'triangle',width:'med',length:'med'}}:{})});
 connector.bringToFront();
}
function note(s,text){s.speakerNotes.textFrame.setText(text);}
function unet(s,x,y,scale=1){
 // Each pair of native rectangles represents a DoubleConv block.
 const coords=[[0,0,96,64],[70,55,77,128],[140,110,58,256],[210,165,42,512],[280,110,58,256],[350,55,77,128],[420,0,96,64]];
 const blocks=[];
 for(let i=0;i<coords.length;i++){
  const [dx,dy,h,ch]=coords[i];
  const b=shape(s,x+dx*scale,y+dy*scale,15*scale,h*scale,{fill:i<4?C.rose:C.ice,line:i<4?C.red:C.blue,geometry:'rect'});
  shape(s,x+(dx+19)*scale,y+dy*scale,15*scale,h*scale,{fill:i<4?C.rose:C.ice,line:i<4?C.red:C.blue,geometry:'rect'});
  txt(s,x+(dx-8)*scale,y+(dy+h+5)*scale,51*scale,25*scale,String(ch),18*scale);
  blocks.push(b);
 }
 for(let i=0;i<6;i++)link(s,blocks[i],blocks[i+1]);
 for(let i=0;i<3;i++)link(s,blocks[i],blocks[6-i],{dash:true});
 return blocks;
}

// Compact, image-led figures. Extended methods and interpretation are in notebook captions.
const s=ppt.slides.add();s.background.fill='#FFFFFF';
txt(s,25,5,1500,34,'a  Benchmark experiment',25,true,C.ink,'left');
const data=box(s,25,50,440,288,'Sen1Floods11');
await img(s,'training_rgb.png',43,97,192);
await img(s,'training_label.png',255,97,192);
txt(s,43,295,192,28,'Sentinel-2',22);
txt(s,255,295,192,28,'Labels',22);
const model=box(s,515,50,550,288,'RGB / RGB+NIR U-Net');
unet(s,595,108,.86);
const evaluation=box(s,1115,50,460,288,'Test-set evaluation');
txt(s,1135,104,174,32,'Input',22,true,C.ink,'left');
txt(s,1290,100,156,50,'Image-mean\nwater IoU',18,true);
txt(s,1438,104,116,32,'F1',22,true);
txt(s,1135,172,174,32,'RGB',25,false,C.blue,'left');
txt(s,1310,172,116,32,'0.2028',25);
txt(s,1438,172,116,32,'0.5399',25);
txt(s,1135,242,174,32,'RGB+NIR',25,false,C.red,'left');
txt(s,1310,242,116,32,'0.5150',25);
txt(s,1438,242,116,32,'0.8404',25);
link(s,data,model);link(s,model,evaluation);

txt(s,25,359,580,34,'b  NYC application',25,true,C.ink,'left');
const nyc=box(s,25,406,440,278,'Sentinel-2 L1C');
await img(s,'nyc_rgb.png',43,453,192);
await img(s,'nyc_nir.png',255,453,192);
txt(s,43,650,192,28,'RGB',22);
txt(s,255,650,192,28,'NIR (B8)',22);
const infer=box(s,515,406,550,278,'Water predictions');
await img(s,'nyc_rgb_mask.png',568,454,194);
await img(s,'nyc_rgb_nir_mask.png',810,454,194);
txt(s,568,650,202,28,'RGB',22);
txt(s,810,650,202,28,'RGB+NIR',22);
const ground=box(s,1115,406,460,278,'FloodNet point comparison');
txt(s,1135,458,420,30,'Detected / observed wet site–date pairs',21);
txt(s,1140,522,230,35,'RGB',25,false,C.blue,'left');
txt(s,1390,522,150,35,'1 / 13',29,true);
txt(s,1140,594,230,35,'RGB+NIR',25,false,C.red,'left');
txt(s,1390,594,150,35,'9 / 13',29,true);
link(s,nyc,infer);link(s,infer,ground);
link(s,model,infer,{from:'bottom',to:'top'});
txt(s,810,357,260,30,'Selected weights',21,false,C.line,'left');
note(s,'Layout inspired by WorldFloods (2021), https://doi.org/10.1038/s41598-021-86650-z, and Portalés-Julià et al. (2023), Fig.5, https://doi.org/10.1038/s41598-023-47595-7. Original redrawing with actual project data. Sen1Floods11 contains 446 pairs with train/validation/test/Bolivia splits 252/89/90/15. RGB: B4/B3/B2; RGB+NIR adds B8. TRAIN-derived channel standardization. Separate models, identical architecture and protocol. Minimum validation Dice loss selects epoch 18 for both. Image-mean water IoU averages water IoU across images. Metrics transcribed from actual notebook08 results. NYC example: Davenport, 21 September 2024. Point counts cover 13 wet site-date pairs on 21 September and 18 October 2024 and are not pixel accuracy. Water masks include permanent water.');

const t=ppt.slides.add();t.background.fill='#FFFFFF';
const supervision=box(t,25,25,410,230,'Training pair');
await img(t,'training_rgb.png',50,70,162);
await img(t,'training_label.png',248,70,162);
const loss=box(t,515,82,450,116,'Loss');
txt(t,535,128,410,40,'0.5 weighted CE + 0.5 Dice',25);
link(t,supervision,loss,{dash:true});
txt(t,437,80,77,27,'Labels',18,false,C.line);
// Separate training-input and loss anchors avoid overlapping branches at the model top.
const trainAnchor=shape(t,624,318,2,2,{fill:'none',line:'none',geometry:'rect'});
const lossAnchor=shape(t,899,318,2,2,{fill:'none',line:'none',geometry:'rect'});
const turn1=shape(t,229,287,2,2,{fill:'none',line:'none',geometry:'rect'});
const turn2=shape(t,624,287,2,2,{fill:'none',line:'none',geometry:'rect'});
link(t,supervision,turn1,{from:'bottom',to:'top',dash:true,arrow:false});
link(t,turn1,turn2,{dash:true,arrow:false});
link(t,turn2,trainAnchor,{from:'bottom',to:'top',dash:true});
link(t,loss,lossAnchor,{from:'bottom',to:'top',dash:true,kind:'elbow',both:true});
txt(t,300,257,210,27,'Training input',21,false,C.line);
txt(t,921,218,250,29,'Logits / gradients',21,false,C.line,'left');
txt(t,1085,74,480,42,'Solid: inference',23,false,C.line,'left');
txt(t,1085,117,480,42,'Dashed: training or skip links',23,false,C.line,'left');

const input=box(t,25,320,410,350,'Sentinel-2 input');
await img(t,'nyc_rgb.png',40,374,180);
await img(t,'nyc_nir.png',240,374,180);
txt(t,40,562,180,31,'RGB',23);
txt(t,240,562,180,31,'NIR (B8)',23);
txt(t,40,622,380,30,'3 or 4 standardized bands',22);
const network=box(t,495,320,480,350,'U-Net');
unet(t,535,389,.89);
txt(t,518,622,434,30,'Encoder / decoder + skip links',22);
const probabilities=box(t,1035,320,250,350,'Water probability');
await img(t,'nyc_rgb_nir_probability.png',1045,378,230);
txt(t,1040,622,240,30,'Softmax: P(water)',22);
const prediction=box(t,1325,320,250,350,'Water mask');
await img(t,'nyc_rgb_nir_mask.png',1335,378,230);
txt(t,1335,622,230,30,'Argmax',22);
link(t,input,network);link(t,network,probabilities);link(t,probabilities,prediction);
note(t,'Original project-method diagram inspired by Portalés-Julià et al. (2023), Fig.5, https://doi.org/10.1038/s41598-023-47595-7. Training example: India_285297 (Sen1Floods11). Lower inference example: Davenport, 21 September 2024. RGB+NIR probability and mask are actual returned outputs. U-Net encoder widths 64/128/256/512; three max-pooling stages, bilinear upsampling, no cloud head. Softmax over water and non-water, then argmax. Weighted CE and Dice use valid labeled pixels. TRAIN-derived normalization and independently trained RGB/RGB+NIR weights. Arrows terminate at separate model input and supervision anchors for clarity.');

await fs.mkdir(out,{recursive:true});
const draft=path.join(out,'nyc_method_figures.pptx');
await (await PresentationFile.exportPptx(ppt)).save(draft);
for(const [i,slide] of ppt.slides.items.entries()){
 const blob=await ppt.export({slide,format:'png',scale:2});
 await fs.writeFile(path.join(out,`figure_${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));
 const layout=await slide.export({format:'layout'});
 await fs.writeFile(path.join(out,`figure_${i+1}.layout.json`),await layout.text());
}
console.log(draft);
