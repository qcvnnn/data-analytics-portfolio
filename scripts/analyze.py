"""Reproducible CSV checks and a separate regression baseline, not a Loginom rerun."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def main():
    output=ROOT/'results'; output.mkdir(exist_ok=True)
    q=pd.read_csv(ROOT/'data/earthquakes.csv')
    # Dataset contains mm/dd/yyyy and a few ISO timestamp values. Parse separately.
    dates=pd.to_datetime(q.Date,format='%m/%d/%Y',errors='coerce',utc=True)
    missing=dates.isna()
    dates.loc[missing]=pd.to_datetime(q.loc[missing,'Date'],format='ISO8601',errors='coerce',utc=True)
    q['year']=dates.dt.year
    yearly=q.groupby('year').size()
    stats={'rows':len(q),'unique_ids':int(q.ID.nunique()),'duplicate_ids':int(q.ID.duplicated().sum()),
           'date_parse_failures':int(dates.isna().sum()),'date_min':str(dates.min().date()),'date_max':str(dates.max().date()),
           'type_counts':q.Type.value_counts().to_dict(),'magnitude_mean':float(q.Magnitude.mean()),
           'magnitude_min':float(q.Magnitude.min()),'magnitude_max':float(q.Magnitude.max()),
           'depth_median':float(q.Depth.median()),'depth_mean':float(q.Depth.mean()),
           'peak_year':int(yearly.idxmax()),'peak_year_count':int(yearly.max()),
           'core_missing':q[['Magnitude','Depth','Latitude','Longitude']].isna().sum().to_dict()}
    (output/'earthquakes-summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    yearly.rename('events').to_csv(output/'earthquakes-by-year.csv')

    houses=pd.read_csv(ROOT/'data/melb_data.csv')
    sales=pd.to_datetime(houses.Date,format='%d/%m/%Y',errors='coerce')
    houses['house_age']=sales.dt.year-houses.YearBuilt
    keep=(houses.house_age.between(0,200)&houses.BuildingArea.between(20,1000)&
          houses.Price.gt(0)&houses.Car.notna()&houses.Landsize.le(10000))
    clean=houses.loc[keep].copy()
    # Price-derived categories are for stratification ONLY, never a predictor.
    clean['stratum']=pd.cut(clean.Price,[-np.inf,650000,1000000,1500000,2000000,np.inf],right=False,labels=False)
    rng=np.random.default_rng(2026); train_ids=[]; test_ids=[]
    for _,group in clean.groupby('stratum'):
        indices=rng.permutation(group.index.to_numpy()); n=round(len(indices)*0.8)
        train_ids.extend(indices[:n]); test_ids.extend(indices[n:])
    train=clean.loc[train_ids]; test=clean.loc[test_ids]
    assert set(train_ids).isdisjoint(test_ids)
    x=np.column_stack([np.ones(len(train)),train.BuildingArea.to_numpy()])
    coef=np.linalg.lstsq(x,train.Price.to_numpy(),rcond=None)[0]
    predictions=coef[0]+coef[1]*test.BuildingArea.to_numpy()
    y=test.Price.to_numpy(); error=y-predictions
    baseline=np.full(len(test),train.Price.mean())
    metrics={'raw_rows':len(houses),'filtered_rows':len(clean),'train_rows':len(train),'test_rows':len(test),
             'seed':2026,'features':['BuildingArea'],'target':'Price','intercept':float(coef[0]),'coefficient':float(coef[1]),
             'test_mae':float(np.abs(error).mean()),'test_rmse':float(np.sqrt(np.mean(error**2))),
             'test_r2':float(1-np.sum(error**2)/np.sum((y-y.mean())**2)),
             'baseline_test_rmse':float(np.sqrt(np.mean((y-baseline)**2))),
             'note':'Independent Python baseline. Different RNG/split from Loginom; not its test metrics.'}
    (output/'melbourne-baseline.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps({'earthquakes':stats,'melbourne':metrics},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
