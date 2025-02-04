import numpy as np
import matplotlib.pyplot as plt

'''
Transfer matrix method

Inputs
c_wvl: central wavelength
fulln: array with all refractive indices
fullw: array with all layer widths

Outputs:
r: reflection coefficient
t: transmission coefficient
x: position
Nn: refractive indices (vs x)
E: optical field (vs x)
'''
def tmm(wvl, fulln, fullw):
    # Exponent factors
    d = fullw*2*np.pi/wvl*fulln

    # Initiate arrays
    x = []
    E = []
    Nn = []
    N = len(fulln)
    M = np.zeros((2,2,N),dtype=complex)
    rs = np.zeros(N)
    ts = np.zeros(N)

    # Loop through layers
    for ii in range(N-1):

        # n of adjacent layers
        n1 = fulln[ii]
        n2 = fulln[ii+1]
        #print("full n1 and n2 are", n1,n2)

        # Fresnel relations
        rs[ii] = (n1 - n2)/(n1+n2)
        ts[ii] = 2*n1/(n1+n2)
        
        # Compose transfer matrix
        M[:,:,ii] = np.dot([[np.exp(-1j*d[ii]),0],
                            [0,np.exp(1j*d[ii])]],
                           [[1, rs[ii]],[rs[ii],1]]) * 1/ts[ii]

        # Multiply with full matrix (if exists)
        if ii >= 1:
            Mt = np.dot(Mt,M[:,:,ii])
        else:
            Mt = M[:,:,0]

    # Reflection and transmission coefficients
    r = Mt[1,0]/Mt[0,0]
    t = 1/Mt[0,0]

    # Initiate arrays
    v1 = np.zeros(len(fullw),dtype=complex)
    v2 = np.zeros(len(fullw),dtype=complex)
    v1[0] = 1
    v2[0] = r

    for ii in range(1,N):
        # Coefficients
        vw = np.linalg.solve(M[:,:,ii-1], [v1[ii-1],v2[ii-1]])
        v1[ii] = vw[0]
        v2[ii] = vw[1]

        # Location array
        xloc = np.arange(0,fullw[ii],5)
       
        #print("xloc:",xloc)

        # Electric fields
        Eloc1 = v1[ii]*np.exp(1j*2*np.pi/wvl*fulln[ii]*xloc)
        Eloc2 = v2[ii]*np.exp(-1j*2*np.pi/wvl*fulln[ii]*xloc)

        # Append to arrays
        x = np.hstack((x,xloc+sum(fullw[:ii])))
        E = np.hstack((E,(Eloc1+Eloc2)))
        Nn = np.hstack((Nn,fulln[ii]+(xloc*0)))
        #print("updated x is:",x)

    #print(x)
    #print("x lenght ::",len(x))
    # Sort arrays()
    ix = np.argsort(x)
    x = x[ix]
    E = E[ix]
    Nn = Nn[ix]
    #print("E is::",E)

    return r, t, x, Nn, E

def plot_results(x,Nn,E,c_wvls,R):
    # Plot n and E
    fig, ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(x,Nn,'b')
    ax[0].set_ylabel('Refractive index n')
    ax[0].set_title('r = %.5f, t = %.5f' %(abs(r),abs(t)))
    ax[1].plot(x,abs(E)**2,'r')
    ax[1].set_ylabel('Normalized |E|^2')
    ax[1].set_xlabel('Distance (um)')
    ax[1].set_xlim([min(x),max(x)])


   # Plot reflectance vs c_wvls
    plt.figure()
    plt.plot(c_wvls,R,'b')
    plt.ylabel('Reflectance (%)')
    plt.xlabel('Wavelength (nm)')
    plt.xlim(min(c_wvls),max(c_wvls))
    plt.ylim(0,110)
    
    plt.show()


def plot_E(x, Nn, E):
    """
    Plot refractive index (Nn) and normalized electric field intensity (|E|^2) on the same plot.

    Parameters:
    - x: array-like, Distance (um).
    - Nn: array-like, Refractive index.
    - E: array-like, Electric field.
    - r: float, Reflection coefficient.
    - t: float, Transmission coefficient.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot refractive index on the left y-axis
    ax1.plot(x, Nn, 'b-', label='Refractive index n')
    ax1.set_ylabel('Refractive index n', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a second y-axis to plot |E|^2
    ax2 = ax1.twinx()
    ax2.plot(x, abs(E)**2, 'r-', label='Normalized |E|^2')
    ax2.set_ylabel('Normalized |E|^2', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Set x-axis label and limits
    ax1.set_xlabel('Distance (um)')
    ax1.set_xlim([min(x), max(x)])

    # Add title and legend
    plt.title(f'r = {abs(r):.5f}, t = {abs(t):.5f}')
    fig.tight_layout()
    plt.show()

###########################
## Example DBR

if __name__ == "__main__":
    
    # central Wavelength (in nm)
    c_wvl = 650
    # Refractive indices
    n1 = 2.58
    n2 = 1.47
    n0 = 1
    ns = 1

    # Number of layers
    Nstk = 4

    # Mirror stack n and width
    Mirrn = np.tile([n1,n2],Nstk)
    #Mirrw = np.tile([c_wvl/(4*n1),c_wvl/(4*n2)],Nstk)
    Mirrw = np.tile([65,110],Nstk)

    # Add air and substrate
    #1.0 instead of 1 makes the array type to have floats as n1 and n2 are float....
    fulln = np.insert([1.0,n0,ns],2,Mirrn)
    #print(Mirrn,(fulln))
    #fullw = np.insert([0,c_wvl/n0,c_wvl/ns],2,Mirrw)
    fullw = np.insert([0,c_wvl,c_wvl],2,Mirrw)
    #print(len(fullw),len(fulln))
    #fullw = np.insert([0,c_wvl,c_wvl],2,Mirrw)
    #print(type(fullw))
    # Run transfer matrix function
    #r,t,x,Nn,E = tmm(c_wvl,fulln, fullw)

    # Units in um and offset
    #x = x/1e3-1

    # Wavelengths array
    wvls = np.linspace(400,1000,601)

    #print(fullw,sum(fullw))

    # Loop through wavelengths
    R = []
    
    r,t,x_length,Nn,E = tmm(wvls[0],fulln, fullw) # for obtaining length of x data
    #print(len(x_length))
    
    E_field=np.zeros((len(wvls),len(x_length)),dtype=complex)
    #E_field=[[]*620]*301
    #E_field=[]

    for ii in range(len(wvls)):
        r,t,x,Nn,E = tmm(wvls[ii],fulln, fullw)
        R.append(abs(r)**2 * 100)
        E_field[:][ii]=E
        #E_field.append(E)

    #print(E_field)
    E_field = np.array(E_field)
    # Units in um and offset
    x = x/1e3-1

    #print(np.shape(E_field))
    

    plt.imshow(abs(E_field)**2, aspect='auto', cmap='jet', 
           extent=[min(x), max(x), min(wvls), max(wvls)], origin='lower')
    plt.colorbar(label="Intensity |E|^2")
    plt.xlabel('Position (um)')
    plt.ylabel('Wavelength (nm)')
    plt.title('Electric Field Intensity')
    plt.show()
    
    plot_results(x,Nn,E_field[135][:],wvls,R)
    plot_E(x,Nn,E_field[135][:])