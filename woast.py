import xarray as xr
import numpy as np
import dask.array as dka
import warnings
warnings.simplefilter(action='ignore', category=dka.PerformanceWarning)


def woast(arr,Ntht,scales,xydim,var=None,normalize=True):
	tmp = WST(arr,Ntht,scales,xydim,var)
	if normalize==True:
		tmp.normalize()
	return tmp.data


class WST():
    """docstring for WSTOA"""
    def __init__(self,arr,Ntht,scales,xydim,var=None):

        # check of arr's type :
        if str(type(arr))   == "<class 'xarray.core.dataset.Dataset'>":
            isda = False
            arr  = arr[var]
        elif str(type(arr)) == "<class 'xarray.core.dataarray.DataArray'>":
            isda = True
        else:
            raise ValueError('arr must be a xarray Dataset or DataArray.')
        
        x,y = [arr[dim] for dim in xydim]

        self.isda   = isda
        self.scales = scales
        self.Ntht   = Ntht
        self.nx     = x.size
        self.ny     = y.size
        self.x      = x
        self.y      = y
        self.xydim  = (y.name,x.name)
        self.build_wavelets()
        self.apply_wavelets(arr,var)

    def fft(self,da):
        dim  = (da.dims.index(self.y.name),da.dims.index(self.x.name))
        func = lambda x: np.fft.fft2(x,axes=dim)
        return xr.apply_ufunc(func,da,dask='allowed')

    def ifft(self,da):
        dim  = (da.dims.index(self.y.name),da.dims.index(self.x.name))
        func = lambda x: np.fft.ifft2(x,axes=dim)
        return xr.apply_ufunc(func,da,dask='allowed')

    def convolve(self,da,wlt):
        dim  = (da.dims.index(self.y.name),da.dims.index(self.x.name))
        func = lambda x,y: abs(np.fft.ifft2(np.fft.fft2(x,axes=dim)*y,axes=dim))
        return xr.apply_ufunc(func,da,wlt,dask='allowed')

    def build_wavelets(self,gamma=1):
        '''
        based on Kymatio and pyWST.
        '''
        if (type(self.scales)==int) or (type(self.scales)==list):
            self.scales = np.array(self.scales)
        if self.scales.size==1:
            self.scales = np.append(self.scales,np.nan)
            cut = 1
        else:
            cut = 0
            
        # Params :
        M, N  = self.ny, self.nx
        theta = -((np.pi/self.Ntht) * np.arange(self.Ntht) + np.pi/2)
        j     = 1
        k0    = 3 * np.pi / 4  # Central wavenumber of the mother wavelet
        k     = k0 / self.scales #2 ** j
        nj    = self.scales.size
        
        # Init :
        gabor = np.zeros((self.Ntht,nj,M,N), complex)
        gauss = np.zeros((self.Ntht,nj,M,N), complex)
        
        R    = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        RInv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        D    = np.array([[1, 0], [0, gamma ** 2]])
        curv = np.array([np.dot(R[:,:,i], np.dot(D, RInv[:,:,i])) for i in range(self.Ntht)])
        curv = curv[:,None]/(2 * self.scales[None,:,None,None] ** 2)
        
        for ex in [-2, -1, 0, 1]:
            for ey in [-2, -1, 0, 1]:
                [xx, yy]  = np.mgrid[ex * M:M + ex * M, ey * N:N + ey * N]
                arg_gabor = -(curv[:,:,0, 0][:,:,None,None] * xx[None,None,:,:] ** 2 + (curv[:,:,0, 1] + curv[:,:,1, 0])[:,:,None,None] * xx[None,None,:,:] * yy[None,None,:,:] + curv[:,:,1, 1][:,:,None,None] * yy[None,None,:,:] ** 2) + 1.j * (xx[None,None,:,:] * (k[None,:] * np.cos(theta[:,None]))[:,:,None,None] + yy[None,None,:,:] * (k[None,:] * np.sin(theta[:,None]))[:,:,None,None])
                arg_gauss = -(curv[:,:,0, 0][:,:,None,None] * xx[None,None,:,:] ** 2 + (curv[:,:,0, 1] + curv[:,:,1, 0])[:,:,None,None] * xx[None,None,:,:] * yy[None,None,:,:] + curv[:,:,1, 1][:,:,None,None] * yy[None,None,:,:] ** 2)
                gabor += np.exp(arg_gabor)
                gauss += np.exp(arg_gauss)
                
        normFactor = 2 * np.pi * self.scales ** 2 / gamma
        gabor /= normFactor[None,:,None,None]
        gauss /= normFactor[None,:,None,None]
        K = np.sum(gabor,axis=(-2,-1)) / np.sum(gauss,axis=(-2,-1))
        morlet = gabor - K[:,:,None,None] * gauss
        
        if cut==1:
            morlet = morlet[:,0]
        
        self.Wlt = xr.DataArray(
            data = morlet / 2**(2*j),
            dims = ["theta1","scale1",*self.xydim],
            coords = {
                    'theta1'    : np.rad2deg((np.pi/self.Ntht) * np.arange(self.Ntht)),
                    'scale1'    : self.scales,
                    self.y.name : self.y,
                    self.x.name : self.x
                      },
            name  = 'Wlt',
            attrs={'description':"Bank of Wavelets.",'long_name':'Wavelets'}
        )

        self.phi0 = xr.DataArray(
            data = np.sqrt(np.outer(np.hanning(self.nx),np.hanning(self.ny))) ,
            dims = [*self.xydim],
            coords = {self.y.name : self.y,self.x.name : self.x},
            name  = 'phi0'
        )


    def apply_wavelets(self,arr,chunk=None):
        FT_Wlt = self.fft(self.Wlt).chunk({'theta1':1,'scale1':1})
        tmp    = self.convolve(arr,FT_Wlt) 
        ds     = xr.Dataset()
        ds     = ds.assign(S1=(tmp*self.phi0).mean(dim=self.xydim))
        ds     = ds.assign(S2=lambda x: (self.convolve(tmp,FT_Wlt.rename({'theta1':'theta2','scale1':'scale2'}))*self.phi0).mean(dim=self.xydim))
        self.data = ds       


    def normalize(self):
        S1  = self.data.S1
        S2  = self.data.S2
        tht = range(self.Ntht) 
        sc  = range(self.data.scale1.size)
        S2n = S2.isel(theta1=tht,scale1=sc)/S1.isel(theta1=tht,scale1=sc)
        self.data = self.data.assign(S2n=S2n)


def closest_Pow2(x):
    i = 1
    while i <= x:
        if i & x:
            tmp = i
        i <<= 1
    n = int(np.log2(tmp)-np.log2(1))
    
    return tmp, n

# operator must be applyable directly on our dataset
# so we have to compute the domain size before to create 
# the banklet of wavelets.

# Chunk the wavelets (??) --> No!!! you dumby dumb
# remove the scales def.
# use of scipy and numpy wont be more efficient with xarray




