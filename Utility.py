# Description: This file contains the utility functions for the project.
# update time: 2024/07/19

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from astropy.coordinates import match_coordinates_sky
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table

import lmfit
from lmfit.lineshapes import gaussian2d
from lmfit.models import LorentzianModel
from lmfit.models import ExpressionModel

def gaussian(x, amp, cen, sig):
    """
    1-d gaussian: gaussian(x, amp, cen, wid)
    
    input: number, array
    """
    return (amp / (np.sqrt(2*np.pi) * sig)) * np.exp(-(x-cen)**2 / (2*sig**2))

def gaussian2d(x, y, amp, cenx, ceny, sigx, sigy):
    """
    2-d gaussian: gaussian(x, y, amp, cenx, ceny, sigx, sigy)
    
    input: number, array
    """
    return (amp / (2*np.pi * sigx*sigy)) * np.exp(-((x-cenx)**2 / (2*sigx**2)) - ((y-ceny)**2 / (2*sigy**2)))

def Poisson(x, k):
    
    return (np.exp(-k)*k**x)/np.math.factorial(x)

def read_data_file(file):
    
    with open(file, 'r') as f:
        lines = f.readlines()
    data = []
    
    for line in lines:
        if line[0] != '#':
            data.append([float(x) for x in line.split(',')])
    data = np.array(data)
    return data

def read_data_file2(file):
    
    with open(file, 'r') as f:
        lines = f.readlines()
    data = []
    
    for line in lines:
        if line[0] != '#':
            data.append([float(x) for x in line.split(' ')])
    # make data a pd.DataFrame
    data = pd.DataFrame(data)
    data.columns = ['x', 'y']
    return data

# pd.read_fwf
def read_txt_file(file):
    
    with open(file, 'r') as f:
        lines = f.readlines()
        
    data = []

    for line in lines:
        if line[0] != '#':
            # Strip the line first to remove trailing newline characters
            clean_line = line.strip()
            # Split the cleaned line and get the last two elements
            split_data = clean_line.split('   ')[-2:]
            print(split_data)

def CrossmatchDisfit(file, cname, fitrange=70, grid=101, weight=1, mode=2):
    
    """
    This function is used to fit the crossmatch distance distribution of two catalogs.
    
    Return: 2D fitting plot
    
    Parameters:
    file: str, the file path of the crossmatch result
    cname: list, the column name of the second catalogs, first catalog is RA and DEC
    fitrange: int, the range of the fitting plot in arcsec
    grid: int, the number of the grid in the fitting plot, need to be odd
    weight: float, the weight of the fitting to the data
    mode: int, the mode of the fitting
        1: 1D fitting
        2: 2D fitting with ra, dec as the x, y axis
        3: 2D fitting with only radius variable
    """

    data = pd.read_csv(file)

    n1 = cname[0]
    n2 = cname[1]

    x = (data.RA-data[n1])*3600*np.cos(data.DEC*np.pi/180)
    y = (data.DEC-data[n2])*3600
    
    if mode == 1:
        
        fig, ax = plt.subplots()
        
        data = {
            'ra': ax.hist(x, grid)[1],
            'rac': ax.hist(x, grid+1)[0],
            'dec': ax.hist(y, grid)[1],
            'decc': ax.hist(y, grid+1)[0]
        }
        
        plt.close(fig)
        
        xedges = np.linspace(-fitrange, fitrange, grid)
        yedges = np.linspace(-fitrange, fitrange, grid)
        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        H = H.T
        z = H.flatten()
        
        X, Y = np.meshgrid(np.linspace(-fitrange, fitrange, grid-1), np.linspace(-fitrange, fitrange, grid-1))
        xf = X.flatten()
        yf = Y.flatten()
        Z = griddata((xf, yf), z, (X, Y), method='linear', fill_value=0)
        vmax = np.nanpercentile(Z, 99.9)

        dframe = pd.DataFrame(data=data)

        model = LorentzianModel()

        paramsx = model.guess(dframe['rac'], x=dframe['ra'])
        paramsy = model.guess(dframe['decc'], x=dframe['dec'])

        resultra = model.fit(dframe['rac'], paramsx, x=dframe['ra'])
        cen1x = resultra.values['center']
        sig1x = resultra.values['sigma']
        resultdec = model.fit(dframe['decc'], paramsy, x=dframe['dec'])
        cen1y = resultdec.values['center']
        sig1y = resultdec.values['sigma']
        
        fitx = model.func(dframe['ra'], **resultra.best_values)
        fity = model.func(dframe['dec'], **resultdec.best_values)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        plt.rcParams.update({'font.size': 15})
        # ax = axs[0]
        # art = ax.pcolor(X, Y, Z, vmin=0, vmax=vmax, shading='auto')
        # plt.colorbar(art, ax=ax, label='z')
        # ell = Ellipse(
        #         (cen1x, cen1y),
        #         width = 3*sig1x,
        #         height = 3*sig1y,
        #         edgecolor = 'w',
        #         facecolor = 'none'
        #     )
        # ax.add_patch(ell)
        # ax.set_title('Histogram of Data')
        # ax.set_xlabel('Delta RA [arcsec]')
        # ax.set_ylabel('Delta DEC [arcsec]')

        ax = axs[0]
        ax.plot(dframe['ra'], fitx, label='fit gaussian')
        ax.plot(dframe['ra'], dframe['rac'], 
                marker='s', markersize=5, ls='', label='data point'
                )
        ax.set_title('Center:{0:5.4f}, 1 Sigma:{1:5.3f}'.format(cen1x, sig1x))
        ax.set_xlabel('Delta RA [arcsec]', fontsize=15)
        ax.set_ylabel('count', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        ax.legend()

        ax = axs[1]
        ax.plot(dframe['dec'], fity, label='fit gaussian')
        ax.plot(dframe['dec'], dframe['decc'], 
                marker='s', markersize=5, ls='', label='data point'
                )
        ax.set_title('Center:{0:5.4f}, 1 Sigma:{1:5.3f}'.format(cen1y, sig1y))
        ax.set_xlabel('Delta DEC [arcsec]', fontsize=15)
        ax.set_ylabel('count', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        ax.legend()
        
        fig.suptitle('AKARI-TSL3 x '+file.split('-')[1][:-6] + '  1D fitting')
        
        plt.show()
    
    if mode == 2:

        xedges = np.linspace(-fitrange, fitrange, grid)
        yedges = np.linspace(-fitrange, fitrange, grid)
        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        H = H.T
        z = H.flatten()

        X, Y = np.meshgrid(np.linspace(-fitrange, fitrange, grid-1), np.linspace(-fitrange, fitrange, grid-1))
        xf = X.flatten()
        yf = Y.flatten()
        
        w = z**weight+0.1
        
        model = Gaussian2dModel()
        params = model.guess(z, xf, yf)
        result = model.fit(z, x=xf, y=yf, params=params, weights=w/10)
        Amp = result.values['amplitude']
        cenx = result.values['centerx']
        sigx = result.values['sigmax']
        ceny = result.values['centery']
        sigy = result.values['sigmay']
        
        Z = griddata((xf, yf), z, (X, Y), method='linear', fill_value=0)
        vmax = np.nanpercentile(Z, 99.9)
        
        fit = model.func(X, Y, **result.best_values)

        Zx = Z[int((grid+1)/2)]
        fitx = fit[int((grid+1)/2)]
        Zy = Z.T[int((grid+1)/2)]
        fity = fit.T[int((grid+1)/2)]

        fig, axs = plt.subplots(2, 2, figsize=(15, 13))
        
        plt.rcParams.update({'font.size': 15})
        # plt.rcParams.update({"tick.labelsize": 13})
        
        ax = axs[0, 0]
        art = ax.pcolor(X, Y, Z, vmin=0, vmax=vmax, shading='auto')
        plt.colorbar(art, ax=ax, label='Data point Density')
        ell = Ellipse(
                (cenx, ceny),
                width = 3*sigx,
                height = 3*sigy,
                edgecolor = 'w',
                facecolor = 'none'
            )
        ax.add_patch(ell)
        ax.set_title('Histogram of Data')
        ax.set_xlabel('ΔRA [arcsec]', fontsize=15)
        ax.set_ylabel('ΔDEC [arcsec]', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)

        ax = axs[0, 1]
        art = ax.pcolor(X, Y, Z-fit, shading='auto')
        plt.colorbar(art, ax=ax, label='Data point Density')
        ax.set_title('Residual')
        ax.set_xlabel('ΔRA [arcsec]', fontsize=15)
        ax.set_ylabel('ΔDEC [arcsec]', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)

        ax = axs[1, 0]
        ax.plot(xedges[:grid-1], fitx, label='fit gaussian')
        ax.plot(xedges[:grid-1], Zx, 
                marker='s', markersize=5, ls='', label='data point'
                )
        ax.set_title('y-axis slice, Center:{0:5.3f}, 1σ:{1:5.2f}'.format(cenx, sigx))
        ax.set_xlabel('ΔRA [arcsec]', fontsize=15)
        ax.set_ylabel('count', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        ax.legend()

        ax = axs[1, 1]
        ax.plot(yedges[:grid-1], fity, label='fit gaussian')
        ax.plot(yedges[:grid-1], Zy,
                marker='s', markersize=5, ls='', label='data point'
                )
        ax.set_title('x-axis slice, Center:{0:5.3f}, 1σ:{1:5.2f}'.format(ceny, sigy))
        ax.set_xlabel('ΔDEC [arcsec]', fontsize=15)
        ax.set_ylabel('count', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        ax.legend()

        fig.suptitle('AKARI-TSL3 x '+file.split('-')[1][:-4]+'  2D fitting')

        plt.show()
        
    if mode == 3:

        xedges = yedges = np.linspace(-fitrange, fitrange, grid)
        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        H = H.T
        z = H.flatten()

        X, Y = np.meshgrid(np.linspace(-fitrange, fitrange, grid-1), np.linspace(-fitrange, fitrange, grid-1))
        xf, yf = X.flatten(), Y.flatten()
        
        model = ExpressionModel(
            'amp*exp(-(x**2 / (2*sig**2)) - (y**2 / (2*sig**2)))',
            independent_vars=['x', 'y']
        )
        params = model.make_params(amp=100, sig=sig=fitrange/100)
        
        w = z**weight
        result = model.fit(z, x=xf, y=yf, params=params, weights=w)
        Sigma = result.params['sig'].value
        print(Sigma)
        
        Z = griddata((xf, yf), z, (X, Y), method='linear', fill_value=0)
        Zx = Z[int((grid+1)/2)]
        Zy = Z.T[int((grid+1)/2)]

        fig, axs=plt.subplots(1, 2, figsize=(15, 5), dpi=100)

        ax = axs[0]
        ax.plot(xedges[:grid-1], Zx, 
            marker='s', markersize=5, ls='', label='data points'
            )
        ax.plot(np.linspace(-fitrange, fitrange, 100),
            model.eval(result.params, x=np.linspace(-fitrange, fitrange, 100), y=0),
            label=f'fit gaussian, $\\sigma$={Sigma:.2f}')
        ax.set_title('y-axis slice')
        ax.set_xlabel('Separation [arcsec]')
        ax.legend()

        ax=axs[1]
        ax.plot(yedges[:grid-1], Zy, 
            marker='s', markersize=5, ls='', label='data points'
            )
        ax.plot(np.linspace(-fitrange, fitrange, 100),
            model.eval(result.params, x=0, y=np.linspace(-fitrange, fitrange, 100)),
            label=f'fit gaussian, $\\sigma$={Sigma:.2f}')
        ax.set_title('x-axis slice')
        ax.set_xlabel('Separation [arcsec]')
        ax.legend()
        
        plt.show()
        return Sigma
        
def CrossmatchDisfit2G(file, cname, fitrange=70, grid=101, weight=1):
    
    """
    This function is used to fit 2 Gaussian to the crossmatch distance distribution of two catalogs.
    
    Return: 2D fitting plot
    
    Parameters:
    file: str, the file path of the crossmatch result
    cname: list, the column name of the second catalogs, first catalog is RA and DEC
    fitrange: int, the range of the fitting plot in arcsec
    grid: int, the number of the grid in the fitting plot, need to be odd
    weight: float, the weight of the fitting to the data
    
    """

    data = pd.read_csv(file)

    n1 = cname[0]
    n2 = cname[1]

    x = (data.RA-data[n1])
    y = (data.DEC-data[n2])

    x = (data.RA-data[n1])*3600*np.cos(data.DEC*np.pi/180)
    y = (data.DEC-data[n2])*3600

    xedges = np.linspace(-fitrange, fitrange, grid)
    yedges = np.linspace(-fitrange, fitrange, grid)
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    H = H.T
    z = H.flatten()

    X, Y = np.meshgrid(np.linspace(-fitrange, fitrange, grid-1), np.linspace(-fitrange, fitrange, grid-1))
    xf = X.flatten()
    yf = Y.flatten()

    w = z**weight+0.1

    model = (lmfit.models.Gaussian2dModel(prefix='g1_')
            +lmfit.models.Gaussian2dModel(prefix='g2_')
            )
    params = model.make_params(
        g1_amplitude=1000,
        g1_center=0,
        g1_sigma=3,
        g2_amplitude=0,
        g2_center=0,
        g2_sigma=20
        )
    result = model.fit(z, x=xf, y=yf, params=params, weights=w/10)
      
    Amp1 = result.best_values['g1_amplitude']
    cenx1 = result.best_values['g1_centerx']
    sigx1 = result.best_values['g1_sigmax']
    ceny1 = result.best_values['g1_centery']
    sigy1 = result.best_values['g1_sigmay']
    
    Amp2 = result.best_values['g2_amplitude']
    cenx2 = result.best_values['g2_centerx']
    sigx2 = result.best_values['g2_sigmax']
    ceny2 = result.best_values['g2_centery']
    sigy2 = result.best_values['g2_sigmay']
    
    if (sigx1 < sigx2) and (sigx1 > 2):
        Ampm = Amp1
        cenxm = cenx1
        cenym = ceny1
        sigxm = sigx1
        sigym = sigy1
        Ampe = Amp2
        cenxe = cenx2
        cenye = ceny2
        sigxe = sigx2
        sigye = sigy2
        
    else:
        Ampm = Amp2
        cenxm = cenx2
        cenym = ceny2
        sigxm = sigx2
        sigym = sigy2
        Ampe = Amp1
        cenxe = cenx1
        cenye = ceny1
        sigxe = sigx1
        sigye = sigy1

    # print(cenxm, sigxm, cenxe, sigxe)
    
    Z = griddata((xf, yf), z, (X, Y), method='linear', fill_value=0)
    vmax = np.nanpercentile(Z, 99.9)

    fit = model.func(X, Y, **result.best_values)

    Zx = Z[int((grid+1)/2)]
    Zy = Z.T[int((grid+1)/2)]
    
    edges = np.linspace(-fitrange, fitrange, 5*grid)
    fitxm = gaussian2d(edges, 0, Ampm, cenxm, cenym, sigxm, sigym) 
    fitym = gaussian2d(edges, 0, Ampm, cenxm, cenym, sigxm, sigym)
    fitxe = gaussian2d(0, edges, Ampe, cenxe, cenye, sigxe, sigye)
    fitye = gaussian2d(0, edges, Ampe, cenxe, cenye, sigxe, sigye)

    fig, axs = plt.subplots(2, 2, layout="constrained", figsize=(14, 10))

    plt.rcParams.update({'font.size': 15})
    # plt.rcParams.update({"tick.labelsize": 13})

    ax = axs[0, 0]
    art = ax.pcolor(X, Y, Z, vmin=0, vmax=vmax, shading='auto')
    plt.colorbar(art, ax=ax, label='Data point Density')
    ell = Ellipse(
            (cenxm, cenym),
            width = 3*sigxm,
            height = 3*sigym,
            edgecolor = 'w',
            facecolor = 'none'
        )
    ax.add_patch(ell)
    ax.set_title('Histogram of Data')
    ax.set_xlabel('ΔRA [arcsec]', fontsize=15)
    ax.set_ylabel('ΔDEC [arcsec]', fontsize=15)
    ax.tick_params(axis='both', labelsize=13)

    # ax = axs[0, 1]
    # art = ax.pcolor(X, Y, Z-fit, shading='auto')
    # plt.colorbar(art, ax=ax, label='Data point Density')
    # ax.set_title('Residual')
    # ax.set_xlabel('ΔRA [arcsec]', fontsize=15)
    # ax.set_ylabel('ΔDEC [arcsec]', fontsize=15)
    # ax.tick_params(axis='both', labelsize=13)

    ax = axs[1, 0]
    ax.plot(edges, fitxm, label='matched', lw=3, ls='--')
    ax.plot(edges, fitxe, label='fake-matched?')
    ax.plot(edges, fitxm+fitxe, label='total')
    ax.plot(xedges[:grid-1], Zx, marker='s', markersize=5, ls='', label='data point')
    ax.set_title(
        'y-axis slice, \n Center:{0:5.3f}, 1σ:{1:5.2f}'.format(cenxm, sigxm) + 
        ', Center:{0:5.3f}, 1σ:{1:5.2f}'.format(cenxe, sigxe)
    )
    ax.set_xlabel('ΔRA [arcsec]', fontsize=15)
    ax.set_ylabel('count', fontsize=15)
    ax.tick_params(axis='both', labelsize=13)
    ax.legend()

    ax = axs[1, 1]
    ax.plot(edges, fitym, label='matched', lw=3, ls='--')
    ax.plot(edges, fitye, label='fake-matched?')
    ax.plot(edges, fitym+fitye, label='total')
    ax.plot(yedges[:grid-1], Zy, marker='s', markersize=5, ls='', label='data point')
    ax.set_title(
        'x-axis slice, \n Center:{0:5.3f}, 1σ:{1:5.2f}'.format(cenym, sigym) + 
        ', Center:{0:5.3f}, 1σ:{1:5.2f}'.format(cenye, sigye)
    )
    ax.set_xlabel('ΔDEC [arcsec]', fontsize=15)
    ax.set_ylabel('count', fontsize=15)
    ax.tick_params(axis='both', labelsize=13)
    ax.legend()

    fig.suptitle('AKARI-TSL3 x '+file.split('-')[1][:-6]+'  2D fitting')

    plt.show()

def downsample(data, factor):
    
    """_summary_
    
    This function downsamples and average the data by a factor of `factor`.
    
    Returns:
        np.array:
            The downsampled data.
        
    Args:
        data (array):
            The array you want to downsample.
        factor (int): 
            The factor by which you want to downsample the data.
    """
    
    length = len(data) - len(data)%factor
    
    _sum = np.zeros(int(length/factor))
    
    for i in range(factor):
        _sum = np.array(data[i:length:factor]) + _sum
    
    return _sum/factor

def resample(model, _filter, norm=False):
    
    """ Resample model to the filter wavelength 
    
    Returns:
    --------
    _filter['x'] : array-like
        The filter x values
    
    y : array-like
        The resampled model y values
    
    Parameters:
    -----------
    model : dict
        The model data
    _filter : dict
        The filter data
    """
    
    # interpolate model to the filter wavelength
    f = interp1d(model['x'], model['y'], kind='linear', fill_value="extrapolate")
    y = f(_filter['x'])
    
    if norm:
        # calculate the integral of the filter
        integral = simps(_filter['y'], _filter['x'])
        
        # calculate the integral of the model
        integral_model = simps(y, _filter['x'])
        
        # normalize the model
        y = y * integral / integral_model
    
    return _filter['x'], y

def fnu2flumbda(wl, fnu):
    
    fnu = fnu * u.erg / u.s / u.Hz / u.cm**2
    flu = fnu.to(u.erg / u.s / u.AA / u.cm**2,
            equivalencies=u.spectral_density(wl * u.AA))
    
    return flu.value

def flumbda2fnu(wl, flu):
    
    flu = flu * u.erg / u.s / u.AA / u.cm**2
    fnu = flu.to(u.erg / u.s / u.Hz / u.cm**2,
            equivalencies=u.spectral_density(wl * u.AA))
    
    return fnu.value

def crossmatch(data1, data2, radius=1, merge=True):
    
    ''' Crossmatch two datasets using astropy.coordinates.match_coordinates_sky
    and return a merged dataset with matched rows from both datasets.
    
    Parameters:
    data1 (DataFrame): First dataset to crossmatch, ['ra', 'dec']
    data2 (DataFrame): Second dataset to crossmatch, ['ra', 'dec']
    radius (float): Maximum separation between matches in arcseconds
    
    Returns:
    merged_data (DataFrame): Merged dataset containing matched rows from data1 and data2
    
    ''' 
    
    from astropy.coordinates import match_coordinates_sky
    from astropy.coordinates import SkyCoord
        
    # Create SkyCoord objects for both datasets
    coords1 = SkyCoord(ra=data1['ra'], dec=data1['dec'], unit="deg")
    coords2 = SkyCoord(ra=data2['ra'], dec=data2['dec'], unit="deg")

    # Find the nearest neighbors in coords2 for each point in coords1
    idx, d2d, d3d = match_coordinates_sky(coords1, coords2, nthneighbor=1)

    # Create a mask for matches within the specified radius
    mask = d2d.arcsec <= radius

    # Add matching indices and distances to data1
    data1['idx'] = idx
    data1['d2d'] = d2d.arcsec
    
    if merge:

        # Filter data1 and data2 to only include matches, iloc is used to keep the same index
        matched_data1 = data1[mask].reset_index(drop=True)
        matched_data2 = data2.iloc[idx[mask]].reset_index(drop=True)

        # Merge matched rows from data1 and data2
        merged_data = pd.concat([matched_data1, matched_data2], axis=1)

        return merged_data
    
    else:
        return data1[mask]

def autocorrelation(data, radius=1):
    
    ''' Calculate the autocorrelation of a dataset by crossmatching it with itself.
    
    Parameters:
    data (DataFrame): Dataset to calculate autocorrelation of
    radius (float): Maximum separation between matches in arcseconds
    
    Returns:
    x (DataFrame): list of matched sources
    '''
    
    coords = SkyCoord(ra=data['ALPHA_J2000_F115W'], dec=data['DELTA_J2000_F115W'], unit="deg")
    idx, d2d, d3d = match_coordinates_sky(coords, coords, nthneighbor=2)
    
    mask = d2d.arcsec <= radius

    data['idx'] = idx
    data['d2d'] = d2d.arcsec
    
    x = pd.DataFrame({'auidx': data['idx'], 'aud2d': data['d2d']})
        
    return x.iloc[idx[mask]]

def fk5_to_galactic(ra_list, dec_list):
    from astropy.coordinates import SkyCoord
    from astropy.coordinates import Galactic
    c = SkyCoord(ra=ra_list*u.degree, dec=dec_list*u.degree, frame='fk5')
    c = c.transform_to(Galactic)
    return c.l.degree, c.b.degree

def galactic_to_fk5(l_list, b_list):
    from astropy.coordinates import SkyCoord
    from astropy.coordinates import FK5
    c = SkyCoord(l=l_list*u.degree, b=b_list*u.degree, frame='galactic')
    c = c.transform_to(FK5)
    return c.ra.degree, c.dec.degree

def remove_galactic(data, deg=10, cname=['RA', 'DEC']):
    
    from astropy.coordinates import Longitude
    from astropy.coordinates import Latitude
    
    data['l'], data['b'] = fk5_to_galactic(data[cname[0]].values, data[cname[1]].values)
    data['l'] = Longitude(data['l'], unit=u.degree)
    data['b'] = Latitude(data['b'], unit=u.degree)
    
    data_nogalacticplane = data.loc[bsc['b'].apply(lambda x: x.degree) > deg]  
    
    return data_nogalacticplane

def AB_2_Jy(AB_mag): 
    F = 10**(23-(AB_mag+48.6)/2.5) # [Jy]
    return F

def z2D(z, H0=70, unit=u.pc):
    H = H0 * u.km / u.s / u.Mpc
    c = 3e5 * u.km / u.s
    
    def calculate_distance(z_value):
        D = z_value * c / H
        return D.to(unit)
    
    if isinstance(z, pd.Series):
        return z.apply(calculate_distance)
    else:
        return calculate_distance(z)

def SFR2Halpha(SFR, z):
    
    # Kennicutt 1998
    # SFR [M_Sun/yr] = L_Halpha / 1.26e41[erg/s]
    
    L_Halpha = 1.26e41 * SFR * u.erg / u.s # [erg/s]
    E_Halpha = L_Halpha / (4*np.pi * z2D(z, unit=u.cm)**2) # [erg/s/cm^2]
    
    return E_Halpha

