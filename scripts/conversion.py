
import numpy as np
import pandas as pd




def create_pnet_inputs(jets,jet_data, all_coords=True):
    eps = 1e-32
    """ Creates the input for the PNet model
    """
    eta_rel = jets[...,0]
    phi_rel = jets[...,1]
    pt_rel = jets[...,2]
    mask = jets[...,3]

    eta_jet = jet_data['eta'].values.reshape(-1,1)
    pt_jet = jet_data['pt'].values.reshape(-1,1)




    if all_coords:
        pnet_inputs = jets.copy()

        log_particle_pt = np.log(pt_rel * pt_jet+eps)
        assert log_particle_pt.shape == pt_rel.shape

        log_pt_rel = np.log(pt_rel+eps)

        delta_R = np.sqrt(eta_rel**2 + phi_rel**2)

        epxpypz = etaphipt_rel_to_epxpypz(jets,jet_data)
        e = epxpypz[...,0]
        print(np.min(e))
        print(np.min(pt_rel))
        assert e.shape == pt_rel.shape
        jet_e = np.sum(e,axis=1).reshape(-1,1)
        assert jet_e.shape == (len(jet_data),1)

        log_e = np.log(e+eps)
        log_rel_e = np.log(e/jet_e+eps)

        pnet_inputs = np.dstack((pnet_inputs[...,0],pnet_inputs[...,1],log_pt_rel,\
                                 log_particle_pt,delta_R,log_e,log_rel_e,pnet_inputs[...,1]))
        assert pnet_inputs.shape[-1] == 8
        assert np.all(pnet_inputs[...,0:2] == jets[...,0:2])==True
        print(pnet_inputs[...,7])
        return pnet_inputs

    else:
        pnet_inputs = jets.copy()
    
    return pnet_inputs



def polar_rel_to_polar(jets,jet_data):

    """Converts relative polar coordinates to absolute polar coordinates
    """
    polar_jet  = jets.copy()
    #print('polar jets shape:  ',polar_jet.shape)
    eta_rel = jets[...,0]
    pt_rel = jets[...,2]

    eta_jet = jet_data['eta'].values.reshape(-1,1)
    pt_jet = jet_data['pt'].values.reshape(-1,1)

    eta = eta_rel + eta_jet
    pt = pt_rel*pt_jet




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