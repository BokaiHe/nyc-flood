"""Export the English experiment notebooks; optionally refresh local analysis only."""
from pathlib import Path
import argparse
import re
import sys
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]
(ROOT / 'exports').mkdir(exist_ok=True)
LOCAL_ANALYSIS={'01','10','12'}
CSS='''<style>
:root {--jp-content-font-family:Arial,sans-serif;--jp-ui-font-family:Arial,sans-serif;}
body{background:white!important;color:#171717;}
main{max-width:1100px;margin:20px auto;padding:24px 34px!important;}
.jp-RenderedHTMLCommon{font-size:15px;line-height:1.65;}
.jp-RenderedHTMLCommon h1{font-size:29px;line-height:1.25;border-bottom:1.5px solid #222;padding-bottom:15px;}
.jp-RenderedHTMLCommon h2{font-size:22px;margin-top:28px;border-bottom:1px solid #aaa;padding-bottom:7px;}
.jp-RenderedHTMLCommon table{width:100%;border-collapse:collapse;border-top:1.5px solid #222;border-bottom:1.5px solid #222;font-size:13px;}
.jp-RenderedHTMLCommon th{background:white;border-bottom:1px solid #888;text-align:left;padding:7px;}
.jp-RenderedHTMLCommon td{border:none;text-align:left;padding:6px 7px;}
.jp-RenderedHTMLCommon tr{background:white!important;}
.jp-RenderedImage img{max-width:100%;height:auto;}
.jp-InputPrompt,.jp-OutputPrompt{display:none;}
.jp-OutputArea-output{overflow-x:auto;}
@media(max-width:760px){main{padding:12px!important;margin:0;}}
</style>'''


def export_notebooks(execute_local=False):
    for path in sorted((ROOT/'notebooks').glob('[0-1][0-9]_*.ipynb')):
        number=path.name[:2]
        if number=='00':continue
        nb=nbformat.read(path,4)
        if execute_local and number in LOCAL_ANALYSIS:
            print('Refreshing local analysis:',path.name,flush=True)
            km=KernelManager(kernel_name='python3')
            km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
            NotebookClient(nb,km=km,timeout=180,resources={'metadata':{'path':str(ROOT)}}).execute()
            nbformat.write(nb,path)
        title=nb.cells[0].source.splitlines()[0].lstrip('# ')
        body,_=HTMLExporter(exclude_input=True,exclude_input_prompt=True,exclude_output_prompt=True).from_notebook_node(nb)
        body=re.sub(r'<script\b[^>]*>[\s\S]*?</script>','',body,flags=re.IGNORECASE)
        body=body.replace('</head>',CSS+'</head>').replace('<title>Notebook</title>',f'<title>{title}</title>')
        body=re.sub(r'href="(\d\d_[^"/]+\.ipynb)"',r'href="../notebooks/\1"',body)
        (ROOT/'exports'/f'{path.stem}.html').write_text(body,encoding='utf-8')
        print('Exported:',path.stem,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute-local',action='store_true',help='Refresh 01, 10 and 12 only; never train or run model inference.')
    export_notebooks(parser.parse_args().execute_local)
