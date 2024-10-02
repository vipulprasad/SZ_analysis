
def sz_forecast(ysz, tkev, v_pec, tcmb, z, centerfreq, relcorr, nu_GHz, const, bandf, typ, temp = True):
    #Funzione equivalente al file sz_forecast_bis_filt_type_ysz_no70.pro, ma con un solo tipo di filtro
    #Function to evaluate the band integrated sz spectrum (currently only for Planck bands)
    from numpy import zeros, exp
    from scipy.io import readsav

    kb=1.380650e-23
    h=6.62607e-34
    c=2.99792e10
    cn=c*1e-9
    hc=h*c
    freq_step = (nu_GHz[1]-nu_GHz[0])#*(1+z) #*1e9

    sz_spec = rel_corr_batch_bis(ysz, tkev, v_pec, tcmb, nu_GHz*(1+z), const, relcorr, typ, temp = temp)
    sz_spec_band_int = zeros(len(centerfreq))

    bands = readsav(bandf, python_dict=True) # bandf = "./pl_bandpass_2013.sav" 
    filter = bands['filter_planck'][:,0:len(sz_spec_band_int)]
    #filter = np.exp(-(nu_GHz-centerfreq[i])**2/2./(bw[i]/2.35)**2.)

    if temp:
        # Output in millikelvin
        for i in range(len(centerfreq)):
            sz_spec_band_int[i] = (sz_spec*filter[:,i]).sum()/filter[:,i].sum()
        return sz_spec_band_int
    else:
        # Output in MJy/sr
        for i in range(len(centerfreq)):
            sz_spec_band_int[i] = (sz_spec*filter[:,i]).sum()/filter[:,i].sum()
            # *freq_step
        return sz_spec_band_int
 

def rel_corr_batch_bis(ysz, tkev, v_pec, tcmb, freq_Ghz, const, relcorr, typ, temp = True):
    #Funzione equivalente al file rel_corr_batch_kincorr_bis_ysz.pro
    # Functio to evaluate the sz spectrum
    theta=tkev/const['me']
    beta=v_pec/(const['c']*1e-5)
    relcorr = rel_corr_batch_init(tcmb,freq_Ghz, const)  
    fac1=relcorr['y0']+theta*relcorr['y1']+theta**(2.)*relcorr['y2']+theta**(3.)*relcorr['y3']+theta**(4.)*relcorr['y4']
    fac2=1.*relcorr['y0']/3.+theta*(5.*relcorr['y0']/6.+2.*relcorr['y1']/3.)+theta**2.*(5.*relcorr['y0']/8.+3.*relcorr['y1']/2.+relcorr['y2'])+theta**3.*(-5.*relcorr['y0']/8.+5.*relcorr['y1']/4.+5.*relcorr['y2']/2.+4.*relcorr['y3']/3.)
    fac00 = 1.+theta*relcorr['C1']+theta**2*relcorr['C2']+theta**3.*relcorr['C3']+theta**4.*relcorr['C4']
    fac01 = relcorr['D0']+theta*relcorr['D1']+theta**2.*relcorr['D2']+theta**3.*relcorr['D3']
    
    if typ==6: #Solo SZ termico
        g_rel=relcorr['f1']*relcorr['y0']
    elif typ==1: #SZ termico + Relativistico
        g_rel=relcorr['f1']*(fac1)
    elif typ==2:
        g_rel=relcorr['f1']*(beta/theta)
    elif typ==3:
        g_rel=relcorr['f1']*((beta**2./theta)*fac2+(beta/theta)*const['co']*fac00+(beta**2./theta)*(3.*const['co']**2.-1.)/2.*fac01)
    elif typ==4:
        g_rel=relcorr['f1']*(fac1+(beta**2./theta)*fac2+(beta/theta)*const['co']*fac00+(beta**2./theta)*(3.*const['co']**2.-1.)/2.*fac01)
    elif typ==5:
        g_rel=relcorr['f1']*(relcorr['y0']+beta/theta)
    elif typ==7:
        g_rel=relcorr['f1']*(fac1+((beta**2./theta)*fac2+(beta/theta)*const['co']*fac00+(beta**2./theta)*(3.*const['co']**2.-1.)/2.*fac01)-(beta/theta))
    elif typ==8:
        g_rel=relcorr['f1']*((fac1+(beta**2./theta)*fac2+(beta/theta)*const['co']*fac00+(beta**2./theta)*(3.*const['co']**2.-1.)/2.*fac01)-(relcorr['y0']+beta/theta))
    elif typ==9:  # SZ termico + Relativistico + SZ Cinetico
        g_rel=relcorr['f1']*(fac1+beta/theta)
        

    
    if temp:
        # output in brightness temperatur mK
        dt_sz=const['t0']*ysz*g_rel/relcorr['f1']*1e3 # mK
        return dt_sz
    else: 
        # output in intensity MJy/sr
        di_sz = ysz*g_rel*1e24*2*(const['kb']*const['t0'])**3/(const['hc']**2) # MJy/sr
        return di_sz

def rel_corr_batch_init(tcmb,freq_GHz, const):
    #Funzione equivalente al file rel_corr_batch_kincorr_init.pro
    import numpy as np
    x=const['h']*freq_GHz*1e9/const['kb']/tcmb
    f1=x**(4.)*np.exp(x)/((np.exp(x)-1.)**(2.))
    hdx=x*(np.exp(x)+1.)/(np.exp(x)-1.)-4.
    
    aidx=x**(3.)/(np.exp(x)-1.) 
    chx=np.cosh(x/2.)
    shx=np.sinh(x/2.)
    xtil=x*(chx/shx)
    s=x/shx  


    y0=xtil-4.

    y1a=-10.+47./2.*xtil-42./5.*xtil**(2.)
    y1b=0.7*xtil**(3.)+s**(2.)*(-21./5.+7./5.*xtil)
    y1=y1a+y1b
    
    y2a=-15/2.+1023./8.*xtil-868./5.*xtil**(2.)
    y2b=329./5.*xtil**(3.)-44./5.*xtil**(4.)
    y2c=11./30.*xtil**(5.)
    y2d=-434./5.+658/5.*xtil-242./5.*xtil**(2.)+143./30.*xtil**(3.)
    y2e=-44./5.+187./60.*xtil
    y2=y2a+y2b+y2c+s**(2.)*y2d+s**(4.)*y2e
    
    y3a=15./2.+2505./8.*xtil-7098./5.*xtil**(2.)
    y3b=1425.3*xtil**(3.)-18594./35.*xtil**(4.)
    y3c=12059./140.*xtil**(5.)-128./21.*xtil**(6.)+16./105.*xtil**(7.)
    y3d1=-709.8+14253/5.*xtil-102267./35.*xtil**(2.)
    y3d2=156767./140.*xtil**(3.)-1216./7.*xtil**(4.)+64./7.*xtil**(5.)
    y3d=s**(2.)*(y3d1+y3d2)
    y3e1=-18594./35.+205003./280.*xtil
    y3e2=-1920./7.*xtil**(2.)+1024./35.*xtil**(3.)
    y3e=s**(4.)*(y3e1+y3e2)
    y3f=s**(6.)*(-544./21.+922./105.*xtil)
    y3=y3a+y3b+y3c+y3d+y3e+y3f
    
    y4a=-135./32.+30375./128.*xtil-6239.1*xtil**(2.)
    y4b=61472.7/4.*xtil**(3.)-12438.9*xtil**(4.)
    y4c=35570.3/8.*xtil**(5.)-16568./21.*xtil**(6.)
    y4d=7516./105.*xtil**(7.)-22./7.*xtil**(8.)+11./210.*xtil**(9.)
    y4e1=-62391./20.+614727./20.*xtil
    y4e2=-1368279./20.*xtil**(2.)+4624139./80.*xtil**(3.)
    y4e3=-157396./7.*xtil**(4.)+30064./7.*xtil**(5.)
    y4e4=-2717./7.*xtil**(6.)+2761./210.*xtil**(7.)
    y4e=s**(2.)*(y4e1+y4e2+y4e3+y4e4)
    y4f1=-12438.9+6046951./160.*xtil
    y4f2=-248520./7.*xtil**(2.)+481024./35.*xtil**(3.)
    y4f3=-15972./7.*xtil**(4.)+18689./140.*xtil**(5.)
    y4f=s**(4.)*(y4f1+y4f2+y4f3)
    y4g1=-70414./21.+465992./105.*xtil
    y4g2=-11792./7.*xtil**(2.)+19778./105.*xtil**(3.)
    y4g=s**(6.)*(y4g1+y4g2)
    y4h=s**(8.)*(-682./7.+7601./210.*xtil)
    y4=y4a+y4b+y4c+y4d+y4e+y4f+y4g+y4h
    
    C1 = 10.-47./5.*xtil+7./5.*xtil**(2.)+7./10.*s**(2.)
    C2 = 25.-111.7*xtil+84.7*xtil**(2.)-18.3*xtil**(3.)+1.1*xtil**(4.)+s**(2.)*(847./20.-183./5.*xtil+121./20.*xtil**(2.))+1.1*s**(4.)
    D0 = -2./3.+11./30.*xtil
    D1 = -4.+12.*xtil-6.*xtil**(2.)+19./30.*xtil**(3.)+s**(2.)*(-3.+19./15.*xtil)
    C3=75./4.+(272.*s**(6.))/105. -(21873.*xtil)/40.+(49161.*xtil**2.)/40.-(27519.*xtil**3)/35.+(6684.*xtil**4.)/35.-(3917.*xtil**5.)/210.+(64.*xtil**6.)/105.+s**(4.)*(6684./35.-(66589.*xtil)/420.+(192.*xtil**2.)/7.)+s**(2.)*(49161./80.-(55038.*xtil)/35.+(36762.*xtil**2.)/35.-(50921.*xtil**3.)/210.+(608.*xtil**4.)/35.)
    C4=-75./4.+(341.*s**(8.))/42.-(10443.*xtil)/8.+(359079.*xtil**2.)/40.-(938811.*xtil**3.)/70.+(261714.*xtil**4.)/35.-(263259.*xtil**5.)/140.+(4772.*xtil**6.)/21.-(1336.*xtil**7.)/105.+(11.*xtil**8.)/42.+s**6.*(20281./21.-(82832.*xtil)/105.+(2948.*xtil**2.)/21.)+s**4.*(261714./35.-(4475403.*xtil)/280.+(71580.*xtil**2.)/7.-(85504.*xtil**3.)/35.+(1331.*xtil**4.)/7.)+s**2.*(359079./80.-(938811.*xtil)/35.+(1439427.*xtil**2.)/35.-(3422367.*xtil**3.)/140.+(45334.*xtil**4.)/7.-(5344.*xtil**5.)/7.+(2717.*xtil**6.)/84.)
    D2=-10.+(542.*xtil)/5.-(843.*xtil**2.)/5.+(10603.*xtil**3.)/140.-(409.*xtil**4.)/35.+(23.*xtil**5.)/42.+s**4*(-409./35.+(391.*xtil)/84.)+s**2.*(-843./10.+(10603.*xtil)/70.-(4499.*xtil**2.)/70.+(299.*xtil**3.)/42.)
    D3=-15./2.+(4929.*xtil)/10.-(39777.*xtil**2.)/20.+(1199897.*xtil**3.)/560.-(4392.*xtil**4.)/5.+(16364.*xtil**5.)/105.-(3764.*xtil**6.)/315.+(101.*xtil**7.)/315.+s**6.*(-15997./315.+(6262.*xtil)/315.)+s**4.*(-4392./5.+(139094.*xtil)/105.-(3764.*xtil**2.)/7.+(6464.*xtil**3.)/105.)+s**2.*(-39777./40.+(1199897.*xtil)/280.-(24156.*xtil**2.)/5.+(212732.*xtil**3.)/105.-(35758.*xtil**4.)/105.+(404.*xtil**5.)/21.)
    relcorr={'f1':f1,'y0':y0,'y1':y1,'y2':y2,'y3':y3,'y4':y4,'C1':C1,'C2':C2,'C3':C3,'C4':C4,'D0':D0,'D1':D1,'D2':D2,'D3':D3}

    return relcorr


def rel_corr_sec_ord_ysz(ysz, yp, tcmb, tkev, freq_GHz=None, temp=True, const=None, secordcorr=None):

    theta = tkev/const["me"]

    if secordcorr is None:
        if freq_GHz is None:

            tcmb = const['t0']
            ni_min = 0.5
            ni_max = 35.0
            step_ni = 0.03
            n_ni = round((ni_max-ni_min)/step_ni)
            ni = np.linspace(ni_min, ni_max, n_ni)
            x = (const['hc']/const['kb'])*(ni/tcmb)

            print('REL_CORR_SEC_ORD - Frequency not supplied, using standard range')
            print('REL_CORR_SEC_ORD - Supplied Tcmb has been ignored.')

        else:
            x = const['h']*freq_GHz*1e9/const['kb']/tcmb

        secordcorr = rel_corr_sec_ord_init(tcmb,freq_GHz,const)


    facn2 = secordcorr['f1']*secordcorr['n2']

    g_cor2=facn2

    if temp:
        dt_sz2 = const['t0']*ysz*(ysz+2.*yp)*g_cor2/secordcorr['f1']*1e3 # in mK arcmin^2 se ysz è in arcmin^2

        return dt_sz2
    else:
        di_sz2 = 2*(const['kb']*const['t0'])**3/const['hc']**2*ysz*(ysz+2.*yp)*g_cor2

        return di_sz2


def rel_corr_sec_ord_init(tcmb,freq_GHz,const):

    # tcmb:		Microwave Background Temperature in K, AT THE REDSHIFT OF THE CLUSTER!!!
    # freq_GHz:	frequency for spectrum calculation in GHz, scalar or vector. MUST BE SCALED AT THE REDSHIFT OF THE CLUSTER!

    # serves to take into account the second-order SZ formula to include
    # the cosmological y term. See Fabbri et al 1978.
    #
    # here as output I want only the second order term, the one at
    # first order is in relcorr, with or without relativistic corrections.
    #
    # I call everything as in Fabbri, but n2 already takes into account the
    # multiplication by x ^ 3, not as in Fabbri where multiplication
    # it is done later
    
    from numpy import exp, cosh, sinh

    x = const['h']*freq_GHz*1e9/const['kb']/tcmb

    f1 = x**(4.)*exp(x)/(exp(x)-1.)**(2.)
    hdx = x*(exp(x)+1.)/(exp(x)-1.)-4.

    aidx = x**(3.)/(exp(x)-1.)                  #  BB without constants
    chx = cosh(x/2.)
    shx = np.sinh(x/2.)
    xtil = x*(chx/shx)
    s = x/shx

    n2a1 = xtil**(3.)

    n2a2 =- 12.*xtil**(2.)

    n2a3 = 34.*xtil

    n2a4 =- 16.

    n2a5 = 2.*x**2/shx**(2.)*(xtil-3.)

    n2 = 1./2.*(n2a1+n2a2+n2a3+n2a4+n2a5)

    secordcorr = {'f1':f1,'n2':n2}

    return secordcorr

def rel_corr_sec_ord_ysz_bndint(ysz, yp, tcmb, tkev, band_centerfreq, freq_GHz=None, temp=True, const=None, secordcorr=None):
    from scipy.io import readsav
    spectrum = rel_corr_sec_ord_ysz(ysz, yp, tcmb, tkev, freq_GHz, temp, const, secordcorr)

    tsz_corrctn_band_int = np.zeros(len(band_centerfreq))
    pl_band=readsav("./pl_bandpass_2013.sav", python_dict=True)
    filt_planck=pl_band['filter_planck'][:,0:len(tsz_corrctn_band_int)]
    #filt=np.exp(-(nu_GHz-centerfreq[i])**2/2./(bw[i]/2.35)**2.)

    for i in range(len(band_centerfreq)):
        tsz_corrctn_band_int[i]=(spectrum*filt_planck[:,i]).sum()/filt_planck[:,i].sum()

    return tsz_corrctn_band_int
