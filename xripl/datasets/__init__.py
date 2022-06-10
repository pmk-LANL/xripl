# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 2022

Submodule/directory with data for examples and testing.

@author: Pawel M. Kozlowski
"""

import os.path as osp
from xripl.reader import openRadiograph

def _fetch(data_filename):
    r"""
    Fetches the absolute path of the file give the relative location
    of the file within the xripl package.
    
    Parameters
    ----------
    data_filename : str
        Relative path of file within the xripl.datasets submodule.
        
    Returns
    -------
    resolved_path : 
        Absolute path to data file within xripl.datasets submodule.
    """
    data_dir = osp.abspath(osp.dirname(__file__))
    resolved_path = osp.join(data_dir, data_filename)
    # check that the resolved path corresponds to a files that exists
    if not osp.isfile(resolved_path):
        raise Exception(f"No such file {resolved_path}")
    return resolved_path


def shot81431():
    r"""
    Gets foreground and background raw image for Marble VC shot 81431 on
    Omega-60.
    
    Returns
    -------
    foreground : numpy.ndarray
        Shot image.
        
    background : numpy.ndarray
        Darkfield (background) image.
    
    """
    # raw data file 
    fileRel = 'xrfccd_xrfc5_t3_81431.h5'
    # getting absolute paths to data files
    fileAbs = _fetch(fileRel)
    # loading the file into a numpy array
    foreground, background = openRadiograph(fileName=fileAbs)
    return foreground, background


def shot86456():
    r"""
    Gets foreground and background raw image for COAX shot 86456 on Omega-60.
    
    Returns
    -------
    foreground : numpy.ndarray
        Shot image.
        
    background : numpy.ndarray
        Darkfield (background) image.
    
    """
    # raw data file 
    fileRel = 'xrfccd_xrfc5_t6_86456.h5'
    # getting absolute paths to data files
    fileAbs = _fetch(fileRel)
    # loading the file into a numpy array
    foreground, background = openRadiograph(fileName=fileAbs)
    return foreground, background