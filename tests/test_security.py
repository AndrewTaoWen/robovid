import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from robovid import ingest


def test_download_input_cannot_be_interpreted_as_yt_dlp_options(monkeypatch,tmp_path):
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=json.dumps({'id':'sample','license':'Creative Commons'}))
    monkeypatch.setattr(ingest.subprocess,'run',run)
    supplied='--exec=touch_unwanted_file'
    ingest.download(supplied,tmp_path)
    assert len(calls)==2
    assert all(args[-2:]==['--',supplied] for args in calls)
