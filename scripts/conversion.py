
import numpy as np
import pandas as pd


def polar_rel_to_polar(jets,jet_data):

    """Converts relative polar coordinates to absolute polar coordinates
    """
    eta_rel = jets[...,0]
    pt_rel = jets[...,2]

    eta_jet = jet_data['eta'].values.reshape(-1,1)
    pt_jet = jet_data['pt'].values.reshape(-1,1)

    eta = eta_rel + eta_jet
    pt = pt_rel*pt_jet



    polar_jet  = jets.copy()

    polar_jet[...,0] = eta
    polar_jet[...,2] = pt

    return polar_jet



def etaphipt_epxpypz(jets):

    """Converts polar coordinates to cartesian coordinates
    """
    eta = jets[...,0]
    phi = jets[...,1]
    pt = jets[...,2]




    px = (pt*np.cos(phi))
    py = (pt*np.sin(phi))
    pz = (pt*np.sinh(eta))
    En = (pt*np.cosh(eta))

    cart_jet = jets.copy()
    cart_jet[...,0] = En
    cart_jet[...,1] = px
    cart_jet[...,2] = py
    cart_jet[...,3] = pz

    
    return cart_jet

def etaphipt_rel_to_epxpypz(jets,jet_data):

    """Converts relative polar coordinates to cartesian coordinates
    """
    polar_jet = jets.copy()
    polar_jet = polar_rel_to_polar(polar_jet,jet_data)
    cart_jet = etaphipt_epxpypz(polar_jet)

    return cart_jet

   



def mjj_jets(jet):
    """Calculates the invariant mass of the jet"""

    
    Ec = np.sum(jet[...,0],axis=1)
    pxc = np.sum(jet[...,1],axis=1)
    pyc = np.sum(jet[...,2],axis=1)
    pzc = np.sum(jet[...,3],axis=1)

    m = np.sqrt(Ec**2-pxc**2-pyc**2-pzc**2)
   
    return m

def pt_jets(jet):
    """Calculates the transverse momentum of the jet"""


    px = np.sum(jet[...,1],axis=1)
    py = np.sum(jet[...,2],axis=1)
    pt = np.sqrt(px**2+py**2)
    return pt