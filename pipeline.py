"""Predict next-season points per game for returning NBA players."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = ['AGE','GP','MIN','PTS','REB','AST','FGA','FTA','FG3A','prior_pts','pts_change']

def load_seasons(start, end, cache):
    from nba_api.stats.endpoints import leaguedashplayerstats
    import time
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in range(start,end+1):
        file = cache / f'{year}.csv'
        if not file.exists():
            season = f'{year-1}-{year%100:02d}'
            frame = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season, season_type_all_star='Regular Season',
                per_mode_detailed='PerGame', timeout=60).get_data_frames()[0]
            frame.to_csv(file,index=False)
            time.sleep(2)
        frame = pd.read_csv(file)
        frame['season'] = year
        frames.append(frame)
    return pd.concat(frames,ignore_index=True)

def prepare(raw):
    required = ['PLAYER_ID','season'] + FEATURES[:9]
    missing = set(required)-set(raw.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    d = raw.copy()
    for col in required:
        d[col] = pd.to_numeric(d[col],errors='raise')
    if d.duplicated(['PLAYER_ID','season']).any():
        raise ValueError('Expected one league-wide row per player-season')
    d = d.sort_values(['PLAYER_ID','season'])
    g = d.groupby('PLAYER_ID')
    prior = g['PTS'].shift(1)
    d['prior_pts'] = prior.where(g['season'].shift(1)==d.season-1)
    d['pts_change'] = d.PTS-d.prior_pts
    d['target'] = g['PTS'].shift(-1)
    d['target_season'] = g['season'].shift(-1)
    d['target_gp'] = g['GP'].shift(-1)
    # Predict PPG conditional on playing at least one game next season.
    return d.loc[(d.target_season==d.season+1)&(d.GP>=20)&(d.target_gp>0)&d.PTS.notna()&d.target.notna()].copy()

def evaluate(d, validation_year, test_year):
    if validation_year>=test_year:
        raise ValueError('Validation must precede test')
    train=d[d.target_season<validation_year]
    val=d[d.target_season==validation_year]
    test=d[d.target_season>=test_year]
    if min(len(train),len(val),len(test))==0:
        raise ValueError('Each chronological split must contain observations')
    def pipe(model):
        return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),model)
    models={'ridge':Ridge(alpha=10),
            'random_forest':RandomForestRegressor(n_estimators=150,min_samples_leaf=8,random_state=42,n_jobs=-1),
            'gradient_boosting':HistGradientBoostingRegressor(max_iter=100,l2_regularization=2,random_state=42)}
    scores={'persistence':float(mean_absolute_error(val.target,val.PTS))}
    for name,model in models.items():
        fitted=pipe(model).fit(train[FEATURES],train.target)
        scores[name]=float(mean_absolute_error(val.target,np.maximum(0,fitted.predict(val[FEATURES]))))
    selected=min(scores,key=scores.get)
    dev=d[d.target_season<test_year]
    pred=test.PTS.to_numpy() if selected=='persistence' else np.maximum(0,pipe(models[selected]).fit(dev[FEATURES],dev.target).predict(test[FEATURES]))
    results={'selected':selected,'validation_mae':scores,'test_mae':float(mean_absolute_error(test.target,pred)),
             'baseline_mae':float(mean_absolute_error(test.target,test.PTS)),
             'test_rmse':float(mean_squared_error(test.target,pred)**.5),
             'baseline_rmse':float(mean_squared_error(test.target,test.PTS)**.5),
             'train_rows':len(train),'validation_rows':len(val),'test_rows':len(test),
             'validation_target_year':validation_year,'first_test_target_year':test_year}
    p=test[['PLAYER_ID','season','target_season','PTS','target']].copy()
    p['prediction']=pred
    return results,p

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',help='Local CSV with NBA API column names and season ending year')
    parser.add_argument('--start',type=int,default=2010)
    parser.add_argument('--end',type=int,default=2025)
    parser.add_argument('--validation',type=int,default=2022)
    parser.add_argument('--test',type=int,default=2023)
    parser.add_argument('--out',default='reports')
    args=parser.parse_args()
    raw=pd.read_csv(args.csv) if args.csv else load_seasons(args.start,args.end,'data/cache')
    metrics,pred=evaluate(prepare(raw),args.validation,args.test)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    pred.to_csv(out/'predictions.csv',index=False)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5))
    ax.scatter(pred.target,pred.prediction,alpha=.35,s=10)
    hi=max(pred.target.max(),pred.prediction.max())
    ax.plot([0,hi],[0,hi],'k--')
    ax.set(xlabel='Actual next-season PPG',ylabel='Predicted next-season PPG',title='Chronological holdout predictions')
    fig.tight_layout();fig.savefig(out/'forecast.png',dpi=150);plt.close(fig)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    main()
