"""Same-address current-visit evidence; physical observations are not BLOCK labels."""
from __future__ import annotations
import numpy as np


def continuity_conflict(points, minimum_jump_km=1.0):
    """Flag kilometre-scale impossible jumps while ignoring sub-km receiver jitter."""
    g=points.sort_values('timestamp',kind='stable').drop_duplicates(['timestamp','latitude','longitude'])
    if len(g)<2:return False
    p=g[['timestamp','latitude','longitude']].to_numpy(float);dt=np.diff(p[:,0])
    lat1,lat2=np.radians(p[:-1,1]),np.radians(p[1:,1]);dlon=np.radians(p[1:,2]-p[:-1,2])
    z=np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    km=6371.0088*2*np.arcsin(np.sqrt(np.clip(z,0,1)))
    speed=np.divide(km*3600/1.852,dt,out=np.zeros_like(km),where=dt>0)
    impossible=((dt==0)&(km>minimum_jump_km))|((dt>0)&(dt<=300)&(km>=minimum_jump_km)&(speed>1200))
    return bool(impossible.any())
