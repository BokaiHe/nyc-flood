"""Paper-style figures from existing observations and model outputs; no model execution."""
from pathlib import Path
import csv
import json
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle, Polygon, ConnectionPatch
from matplotlib.lines import Line2D
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as project

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/paper_figures'
RUN = ROOT / 'outputs/cloud_logs/nyc_inference/nyc_l1c_20260921_013008_456262'
BLUE, ORANGE = '#2684B8', '#E68A2E'
WATER, LAND, INVALID = '#2684B8', '#EEEBDD', '#D1D5DA'
MASK = ListedColormap([INVALID, LAND, WATER])
NORM = BoundaryNorm([-1.5, -.5, .5, 1.5], 3)
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':9,
    'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
    'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,
    'legend.frameon':False,'legend.fontsize':7,'lines.linewidth':1.2,
    'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white'})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.png', dpi=600, bbox_inches='tight', pad_inches=.04)
    fig.savefig(OUT / f'{name}.pdf', dpi=600, bbox_inches='tight', pad_inches=.04)
    fig.savefig(OUT / f'{name}.svg', dpi=600, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)
    return OUT / f'{name}.png'


def title(ax, letter, text):
    ax.set_title(f'({letter}) {text}', loc='left', fontsize=8, fontweight='bold', pad=5)


def image_axis(ax, array, **kwargs):
    ax.imshow(array, interpolation='nearest', **kwargs)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(.55)


def scale(ax, pixels, label, width):
    x, y = width*.06, width*.90
    ax.plot([x,x+pixels],[y,y],color='white',lw=4,solid_capstyle='butt')
    ax.plot([x,x+pixels],[y,y],color='black',lw=1.5,solid_capstyle='butt')
    ax.text(x+pixels/2,y-width*.035,label,ha='center',va='bottom',fontsize=6,
            bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':1})


def stretch(image, limits=None):
    rgb=np.moveaxis(image[:3],0,-1)
    good=np.isfinite(rgb).all(-1)
    if limits is None:
        limits=np.percentile(rgb[good],[2,98],axis=0)
    lo,hi=limits
    shown=np.clip((rgb-lo)/np.maximum(hi-lo,1e-6),0,1)
    shown[~good] = matplotlib.colors.to_rgb(INVALID)
    return np.nan_to_num(shown), limits


def training_data():
    base=ROOT/'data/sen1floods11/v1.1/data/flood_events/HandLabeled'
    with rasterio.open(base/'S2Hand/India_285297_S2Hand.tif') as src:
        x=src.read([4,3,2,8]).astype(float)/10000
    with rasterio.open(base/'LabelHand/India_285297_LabelHand.tif') as src:
        lab=src.read(1)
    rgb,_=stretch(x)
    fig,ax=plt.subplots(1,4,figsize=(7.2,2.05),gridspec_kw={'wspace':.035})
    image_axis(ax[0],rgb);image_axis(ax[1],x[3],cmap='gray',vmin=0,vmax=.6)
    image_axis(ax[2],lab,cmap=MASK,norm=NORM)
    image_axis(ax[3],rgb)
    overlay=np.zeros((*lab.shape,4));overlay[lab==1]=[.149,.518,.722,.7]
    ax[3].imshow(overlay,interpolation='nearest')
    for a,l,t in zip(ax,'abcd',['RGB','NIR (B8)','Human labels','Water overlay']):title(a,l,t)
    fig.subplots_adjust(bottom=.20)
    fig.legend(handles=[Patch(color=WATER,label='Water'),Patch(color=LAND,label='Non-water'),Patch(color=INVALID,label='Ignored')],loc='lower center',ncol=3,bbox_to_anchor=(.5,.01))
    return save(fig,'01_training_data')


def splits():
    p=ROOT/'data/sen1floods11/v1.1/splits/flood_handlabeled'
    names=['train','valid','test','bolivia']; counts=[]
    for n in names:
        with (p/f'flood_{n}_data.csv').open() as f:counts.append(len(list(csv.reader(f))))
    fig,ax=plt.subplots(figsize=(7.2,1.65))
    fig.subplots_adjust(bottom=.48,top=.94)
    left=0
    for n,c,col in zip(['Train','Validation','Test','Bolivia'],counts,[BLUE,'#A9C7DB','#687985','#D9DDE0']):
        ax.barh(0,c,left=left,height=.55,color=col,edgecolor='white',lw=1)
        ax.text(left+c/2,0,str(c),ha='center',va='center',fontsize=8,color='white' if n in ['Train','Test'] else '#111')
        left+=c
    ax.set_xlim(0,sum(counts));ax.set_yticks([]);ax.set_xlabel('Labeled image pairs')
    for spine in ['left','bottom']:ax.spines[spine].set_visible(False)
    fig.legend([Patch(color=c) for c in [BLUE,'#A9C7DB','#687985','#D9DDE0']],['Train','Validation','Test','Bolivia'],ncol=4,loc='lower center',bbox_to_anchor=(.5,.01))
    return save(fig,'02_data_split')


def benchmark(metrics):
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),gridspec_kw={'wspace':.65})
    fig.subplots_adjust(left=.17,right=.99,bottom=.27,top=.88)
    keys=['mean_iou','global_iou','f1','precision','recall']
    for index,split in enumerate(['test','bolivia']):
        ax=axes[index]; y=np.arange(5)
        for j,model in enumerate(['rgb','rgb_nir']):
            r=next(r for r in metrics if r['split']==split and r['model']==model)
            ax.barh(y+(j-.5)*.29,[r[k] for k in keys],height=.26,color=[BLUE,ORANGE][j],label=['RGB','RGB+NIR'][j])
        ax.set_yticks(y,['Image-mean\nwater IoU','Global water IoU','F1','Precision','Recall']);ax.invert_yaxis()
        ax.set_xlim(0,1);ax.set_xticks([0,.25,.5,.75,1]);ax.set_xlabel('Score')
        ax.grid(axis='x',alpha=.15,lw=.5);ax.set_axisbelow(True)
        title(ax,'ab'[index],['Test (90 images)','Bolivia (15 images)'][index])
    fig.legend([Patch(color=BLUE),Patch(color=ORANGE)],['RGB','RGB+NIR'],loc='lower center',ncol=2,bbox_to_anchor=(.5,.01))
    return save(fig,'04_benchmark')


def records():
    return json.loads((ROOT/'data/nyc/l1c_chips/chips.json').read_text())['records']


def read_scene(rec):
    with rasterio.open(ROOT/rec['dn_path']) as src:
        dn=src.read().astype(float);profile=src.profile.copy()
        c,r,w,h=rec['central_window'];tr=src.window_transform(Window(c,r,w,h))
    keep=((dn!=0)&(dn!=65535)&np.isfinite(dn)).all(0)
    ref=(dn+np.array(rec['radio_add_offset'])[:,None,None])/rec['quantification']
    ref[:,~keep]=np.nan
    stem=f"{rec['date']}_{rec['site']['sensor_id']}"
    result={'record':rec,'full':ref,'image':ref[:,r:r+h,c:c+w],'profile':profile,'transform':tr}
    for model in ['rgb','rgb_nir']:
        for field,suffix in [('p','probability'),('m','water_mask')]:
            with rasterio.open(RUN/f'{stem}_{model}_{suffix}.tif') as src:
                assert src.shape==(h,w) and src.transform==tr
                result[f'{model}_{field}']=src.read(1)
    xy=project('EPSG:4326',profile['crs'],[float(rec['site']['longitude'])],[float(rec['site']['latitude'])])
    px,py=(~tr)*(xy[0][0],xy[1][0]);result['point']=(px-.5,py-.5)
    return result


def sensor(ax,point):
    ax.plot(*point,marker='+',ms=6,mew=1,color='#B82833',linestyle='none')


def nyc_plate(recs,name):
    scenes=[read_scene(r) for r in recs]
    nir_cmap=plt.get_cmap('gray').copy();nir_cmap.set_bad(INVALID)
    # Identical display stretch for all selected dates, independent of predictions.
    vals=np.concatenate([np.moveaxis(s['image'][:3],0,-1).reshape(-1,3) for s in scenes])
    limits=np.percentile(vals[np.isfinite(vals).all(1)],[2,98],axis=0)
    fig,axes=plt.subplots(len(scenes),4,figsize=(7.2,1.72*len(scenes)+.2),squeeze=False,
                           gridspec_kw={'wspace':.035,'hspace':.025})
    for i,(s,row) in enumerate(zip(scenes,axes)):
        rgb,_=stretch(s['image'],limits)
        arrays=[rgb,s['image'][3],s['rgb_m'],s['rgb_nir_m']]
        for j,(ax,a) in enumerate(zip(row,arrays)):
            image_axis(ax,a,**({} if j==0 else {'cmap':nir_cmap,'vmin':0,'vmax':.6} if j==1 else {'cmap':MASK,'norm':NORM}))
            sensor(ax,s['point'])
            if j==0:
                scale(ax,40,'400 m',128)
                label=s['record']['date'] if len(scenes)==2 else f"{s['record']['date']}\n{textwrap.fill(s['record']['site']['sensor_name'],width=24)}"
                ax.set_ylabel(label,fontsize=7,labelpad=5)
            if i==0:title(ax,'abcd'[j],['RGB','NIR (B8)','RGB mask','RGB+NIR mask'][j])
    fig.legend([Patch(color=WATER),Patch(color=LAND),Patch(color=INVALID),Line2D([],[],marker='+',color='#B82833',ls='none')],
               ['Water','Non-water','Unavailable','Sensor'],ncol=4,loc='lower center',bbox_to_anchor=(.5,.01))
    return save(fig,name)


def nyc_probabilities(recs):
    fig,axes=plt.subplots(2,2,figsize=(5.6,5.1),gridspec_kw={'wspace':.035,'hspace':.1})
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad(INVALID)
    for i,rec in enumerate(recs):
        s=read_scene(rec)
        for j,model in enumerate(['rgb','rgb_nir']):
            a=axes[i,j];p=np.ma.masked_less(s[f'{model}_p'],0)
            image_axis(a,p,cmap=cmap,vmin=0,vmax=1);sensor(a,s['point'])
            if i==0:title(a,'ab'[j],['RGB probability','RGB+NIR probability'][j])
            if j==0:a.set_ylabel(rec['date'],fontsize=8);scale(a,40,'400 m',128)
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap=cmap),ax=axes.ravel().tolist(),fraction=.028,pad=.025,label='P(water)',ticks=[0,.5,1])
    return save(fig,'07_nyc_probabilities')


def locator(all_records):
    rec=next(r for r in all_records if r['date']=='2024-09-21' and 'davenport' in r['site']['sensor_id'])
    s=read_scene(rec)
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.9),gridspec_kw={'wspace':.14})
    geo=json.loads((ROOT/'data/nyc/cartography/boroughs.geojson').read_text())
    a=axes[0];a.set_facecolor('#E4F1F7')
    for f in geo['features']:
        for polygon in f['geometry']['coordinates']:
            points=np.array(polygon[0]);a.add_patch(Polygon(points,fc='#EDEBE2',ec='#737373',lw=.35))
    unique={r['site']['sensor_id']:r['site'] for r in all_records}
    for sid,site in unique.items():
        a.plot(float(site['longitude']),float(site['latitude']),'o',ms=3,mec='white',mew=.3,color=ORANGE if 'davenport' in sid else BLUE)
    a.plot(float(rec['site']['longitude']),float(rec['site']['latitude']),'o',ms=4,mec='black',mew=.4,color=ORANGE,zorder=10)
    a.set_xlim(-74.27,-73.68);a.set_ylim(40.48,40.93);a.set_aspect(1/np.cos(np.deg2rad(40.7)))
    a.set_xticks([-74.2,-74.0,-73.8],['74.2°W','74.0°W','73.8°W']);a.set_yticks([40.5,40.7,40.9],['40.5°N','40.7°N','40.9°N'])
    a.tick_params(length=2,pad=2,labelsize=6)
    a.annotate('N',xy=(.91,.90),xytext=(.91,.72),xycoords='axes fraction',ha='center',fontsize=7,arrowprops={'arrowstyle':'-|>','lw':.8})
    a.text(.04,.97,'10 sites',transform=a.transAxes,va='top',fontsize=7)
    a.annotate('Davenport',xy=(float(rec['site']['longitude']),float(rec['site']['latitude'])),xytext=(-74.1,40.56),fontsize=6.5,arrowprops={'arrowstyle':'-','lw':.6})
    rgb,_=stretch(s['full']);image_axis(axes[1],rgb);scale(axes[1],100,'1 km',512)
    axes[1].add_patch(Rectangle((191.5,191.5),128,128,fill=False,ec=ORANGE,lw=1))
    image_axis(axes[2],s['rgb_nir_m'],cmap=MASK,norm=NORM);sensor(axes[2],s['point']);scale(axes[2],40,'400 m',128)
    for ax,l,txt in zip(axes,'abc',['NYC sites','5.12 km context','1.28 km prediction']):title(ax,l,txt)
    # Exact central crop extent, with connectors kept outside the image content.
    fig.add_artist(ConnectionPatch((319.5,191.5),(0,1),'data','axes fraction',axesA=axes[1],axesB=axes[2],color='#777',lw=.6))
    fig.add_artist(ConnectionPatch((319.5,319.5),(0,0),'data','axes fraction',axesA=axes[1],axesB=axes[2],color='#777',lw=.6))
    return save(fig,'05_nyc_locator')


def point_comparison(points):
    # Conclusion: more observed wet pairs are detected by RGB+NIR, with variation
    # across dates. Counts summarize this selected sample; every pair stays below.
    dates=list(dict.fromkeys(r['date'] for r in points))
    ordered=[r for date in dates for r in points if r['date']==date]
    groups=[[r for r in ordered if r['date']==date] for date in dates]+[ordered]
    fig=plt.figure(figsize=(7.2,3.35))
    a=fig.add_axes([.10,.56,.86,.34])
    b=fig.add_axes([.10,.19,.79,.15])
    cax=fig.add_axes([.915,.19,.012,.15])
    x=np.arange(len(groups));width=.24
    for model,offset,col,label in [('rgb',-width/2,BLUE,'RGB'),('rgb_nir',width/2,ORANGE,'RGB+NIR')]:
        counts=[sum(int(r[f'{model}_prediction'])==1 for r in group) for group in groups]
        values=counts
        a.bar(x+offset,values,width=width,color=col,zorder=2,label=label)
        for xpos,v,n,group in zip(x+offset,values,counts,groups):
            a.text(xpos,v+.22,f'{n}/{len(group)}',ha='center',va='bottom',fontsize=8)
    a.set_xticks(x,[f'{date[5:]} (n={len(group)})' for date,group in zip(dates,groups)]+[f'All pairs (n={len(ordered)})'])
    a.set_ylim(0,10.5);a.set_yticks([0,2,4,6,8,10]);a.set_ylabel('Detected wet pairs (count)')
    a.set_xlim(-.6,len(groups)-.4);a.grid(axis='y',color='#E8E8E8',lw=.5,zorder=0)
    a.tick_params(axis='x',length=0)
    title(a,'a','Detection at observed wet sites')
    fig.legend(*a.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.01),ncol=2)

    probabilities=np.array([[float(r[f'{model}_probability']) for r in ordered] for model in ['rgb','rgb_nir']])
    heat=b.imshow(probabilities,cmap='Blues',vmin=0,vmax=1,aspect='auto',interpolation='nearest')
    ids=[];start=0
    for date,group in zip(dates,groups):
        ids.extend(f'{date[5:7]}-{j+1}' for j in range(len(group)))
        if start:b.axvline(start-.5,color='white',lw=3)
        start+=len(group)
    for row in range(2):
        for col in range(len(ordered)):
            p=probabilities[row,col]
            model=['rgb','rgb_nir'][row]
            label=f'{p:.2f}'+('*' if int(ordered[col][f'{model}_prediction'])==1 else '')
            b.text(col,row,label,ha='center',va='center',fontsize=6.2,color='white' if p>.60 else '#172E43')
    b.set_xticks(range(len(ordered)),ids,fontsize=6.5)
    b.set_yticks([0,1],['RGB','RGB+NIR']);b.tick_params(length=0)
    b.set_xticks(np.arange(-.5,len(ordered),1),minor=True);b.set_yticks([-.5,.5,1.5],minor=True)
    b.grid(which='minor',color='white',lw=.8);b.tick_params(which='minor',length=0)
    for spine in b.spines.values():spine.set_visible(False)
    title(b,'b','Water probability by site–date pair')
    cb=fig.colorbar(heat,cax=cax,ticks=[0,.5,1]);cb.outline.set_visible(False);cb.ax.tick_params(length=0,labelsize=6)
    b.text(1,1.27,'* Classified as water',transform=b.transAxes,ha='right',fontsize=6.5,color='#444')
    return save(fig,'08_point_comparison')


def archived_panels():
    # Exact rectangular crops of complete existing panels; no digitization or inferred values.
    src=ROOT/'outputs/project_summary/source_images'
    im=plt.imread(src/'training_curves.png')
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.65),gridspec_kw={'wspace':.045})
    for ax,crop,letter,label in zip(axes,[im[39:431,39:611],im[39:431,613:1184]],'ab',
                                   ['Training and validation loss','Validation water IoU']):
        ax.imshow(crop,interpolation='nearest');ax.axis('off');title(ax,letter,label)
    save(fig,'03_training_record')
    im=plt.imread(src/'test_prediction.png')
    fig,axes=plt.subplots(1,5,figsize=(7.2,1.7),gridspec_kw={'wspace':.035})
    bounds=[(41,245),(250,454),(458,662),(667,871),(875,1080)]
    for ax,(left,right),letter,label in zip(axes,bounds,'abcde',['RGB','NIR (B8)','Human labels','RGB mask','RGB+NIR mask']):
        image_axis(ax,im[57:262,left:right]);title(ax,letter,label)
    fig.legend([Patch(color='#167caf'),Patch(color='#eee8d5'),Patch(color='#d7dee7')],
               ['Water','Non-water','Ignored'],ncol=3,loc='lower center',bbox_to_anchor=(.5,.01))
    save(fig,'04b_test_prediction_record')


def build_all(metrics,points):
    paths={'training':training_data(),'splits':splits(),'benchmark':benchmark(metrics)}
    recs=records();d=[r for r in recs if 'davenport' in r['site']['sensor_id']]
    paths['locator']=locator(recs);paths['nyc']=nyc_plate(d,'06_nyc_comparison')
    paths['probability']=nyc_probabilities(d);paths['points']=point_comparison(points)
    for j,start in enumerate(range(0,len(recs),5),1):
        paths[f'gallery{j}']=nyc_plate(recs[start:start+5],f'S{j}_all_nyc_sites')
    archived_panels()
    return paths
