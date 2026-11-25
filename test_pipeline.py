import numpy as np
import pandas as pd
import pytest
from pipeline import prepare,evaluate

def sample():
    rng=np.random.default_rng(42)
    rows=[]
    for player in range(40):
        pts=rng.uniform(4,28)
        for year in range(2015,2025):
            pts=max(1,pts+rng.normal(0,1))
            rows.append(dict(PLAYER_ID=player,season=year,AGE=20+year-2015,GP=60,MIN=25,PTS=pts,REB=4,AST=3,FGA=12,FTA=3,FG3A=4))
    return pd.DataFrame(rows)

def test_alignment():
    d=prepare(sample())
    assert (d.target_season==d.season+1).all()
    assert np.allclose(d.target,d.merge(sample(),left_on=['PLAYER_ID','target_season'],right_on=['PLAYER_ID','season']).PTS_y)
    assert d.loc[d.season==2015,'prior_pts'].isna().all()

def test_missing_season():
    raw=sample()
    raw=raw[~((raw.PLAYER_ID==0)&(raw.season==2018))]
    d=prepare(raw)
    assert not ((d.PLAYER_ID==0)&(d.season==2017)).any()
    assert d.loc[(d.PLAYER_ID==0)&(d.season==2019),'prior_pts'].isna().all()

def test_duplicate():
    raw=sample()
    with pytest.raises(ValueError):
        prepare(pd.concat([raw,raw.iloc[:1]]))

def test_model_smoke():
    metrics,p=evaluate(prepare(sample()),2021,2022)
    assert metrics['test_rows']==120
    assert (p.target_season>=2022).all()
    assert np.isfinite(p.prediction).all()
