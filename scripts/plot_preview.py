"""Create a data preview, explicitly not a Power BI screenshot."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
p=pd.read_csv(ROOT/'data/earthquakes.csv')
d=pd.to_datetime(p.Date,format='%m/%d/%Y',errors='coerce',utc=True)
m=d.isna(); d.loc[m]=pd.to_datetime(p.loc[m,'Date'],format='ISO8601',errors='coerce',utc=True)
p['year']=d.dt.year
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
p.groupby('year').size().plot(ax=ax[0,0],color='#216e96',lw=2)
ax[0,0].set(title='Записи по годам',xlabel='Год',ylabel='Количество записей')
ax[0,1].hist(p.Magnitude,bins=25,color='#5d8eac',edgecolor='white')
ax[0,1].set(title='Распределение магнитуды',xlabel='Магнитуда',ylabel='Количество записей')
ax[1,0].scatter(p.Longitude,p.Latitude,c=p.Depth,s=2,alpha=.4,cmap='viridis',rasterized=True)
ax[1,0].set(title='Координаты событий (не картографическая подложка)',xlabel='Долгота',ylabel='Широта')
counts=p.Type.value_counts(); ax[1,1].barh(counts.index,counts.values,color='#216e96')
ax[1,1].set(title='Типы событий в наборе',xlabel='Количество записей',xscale='log')
fig.suptitle('Землетрясения · проверка исходного CSV\nДополнительная визуализация Python, не скриншот Power BI',fontsize=16)
(ROOT/'figures').mkdir(exist_ok=True)
fig.savefig(ROOT/'figures/earthquakes-preview.png',dpi=150)
plt.close(fig)
