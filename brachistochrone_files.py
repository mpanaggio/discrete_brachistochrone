import numpy as np
from scipy import optimize
def norm(x):
    return np.sqrt(np.sum(x**2))
def angles_with_previous(x,y):
    vec_prev=np.array([x[1]-x[0],y[1]-y[0]])
    vec_next=np.array([1,0])
    vec_prev=vec_prev/norm(vec_prev)
    vec_next=vec_next/norm(vec_next)
    angles=[np.arccos(np.dot(vec_prev,vec_next))]
    for k in range(1,len(x)-1):
        vec_prev=np.array([x[k-1]-x[k],y[k-1]-y[k]])
        vec_next=np.array([x[k+1]-x[k],y[k+1]-y[k]])
        vec_prev=vec_prev/norm(vec_prev)
        vec_next=vec_next/norm(vec_next)
        angles.append(np.arccos(np.dot(vec_prev,vec_next)))
    return angles
def get_optimal_cycloid(a,b):
    """ Compute parameters for cycloid between (0,0) and (a,b)"""
    cycloid=lambda thm,r: np.array([a-(r*(thm-np.sin(thm))),b-(-r*(1-np.cos(thm)))])
    
    #sol=optimize.root(lambda X: cycloid(X[0],X[1]),[np.pi-0.01,1],options={'xtol':1e-8})
#     guess=sol.x
#     print(guess)
#     guess[0]=np.array([guess[0],2*np.pi]).min()
#     sol=optimize.minimize(lambda X: np.sum((cycloid(X[0],X[1]))**2),guess,options={'xtol':1e-8},bounds=[(0,2*np.pi),(0,np.inf)])
    sse=lambda x: np.sum(cycloid(x[0],x[1])**2)
    
    sol=optimize.brute(sse,ranges=[(0,2*np.pi),(0,10)],Ns=20)
    guess=sol
    #print(guess)
    sse_grad=lambda thm,r: np.array([2*(a-(r*(thm-np.sin(thm))))*(-r*(1-np.cos(thm)))+\
                                     2*(b-(-r*(1-np.cos(thm))))*(-r*np.sin(thm)),\
                                     2*(a-(r*(thm-np.sin(thm))))*(-(thm-np.sin(thm)))+\
                                     2*(b-(-r*(1-np.cos(thm))))*(1-np.cos(thm))])
    sse_grad2=lambda x: sse_grad(x[0],x[1])
    sol=optimize.minimize(sse,x0=guess,jac=sse_grad2,method='SLSQP',bounds=[(0,2*np.pi),(0,np.inf)])
    #print(sol)
    theta_end=sol.x[0]
    r=sol.x[1]
    #print(theta_end,r)
    return theta_end,r

def make_guess(a,b,N):
    """ Make a guess for z with points along a cycloid"""
    theta_end,r=get_optimal_cycloid(a,b)
    theta=np.linspace(0,theta_end,N+1)
    x_cyc_guess=r*(theta-np.sin(theta))
    y_cyc_guess=-r*(1-np.cos(theta))
    guess=np.concatenate([np.diff(x_cyc_guess),-y_cyc_guess])
    return guess

def compute_optimal_path(a,b,N):
    """ Solve nonlinear optimization problem for optimal N-segment path between (0,0) and (a,b)"""
    A=np.concatenate([np.ones((1,N)),np.zeros((1,N+1))],axis=1) # sum dx = a
    dx_con=optimize.LinearConstraint(A,a,a)
    bounds=[(0,a) for k in range(N)] + [(0,0)] +[(0,np.inf) for k in range(N-1)]+[(-b,-b)]
    
    f= lambda z: np.sum(np.sqrt(z[:N]**2+(z[N+1:]-z[N:-1])**2)/(np.sqrt(z[N+1:])+np.sqrt(z[N:-1])))
    y_k=lambda z,k: z[N+k] # shorthand for y
    dist_k=lambda z,k: np.sqrt(z[k-1]**2+(y_k(z,k)-y_k(z,k-1))**2) if 1<=k<=N else np.nan # shorthand for distance 

    gradf=lambda z: np.concatenate([ 
        (z[:N]/np.sqrt(z[:N]**2+(z[N+1:]-z[N:-1])**2)/(np.sqrt(z[N+1:])+np.sqrt(z[N:-1]))).reshape(-1), # derivative w.r.t Delta x_k
        [0],    # this derivative is undefined but we set it to zero since y_0 is known and fixed.
        [-(y_k(z,k+1)-y_k(z,k))/(np.sqrt(y_k(z,k+1))+np.sqrt(y_k(z,k)))/dist_k(z,k+1)\
         -dist_k(z,k+1)/2/np.sqrt(y_k(z,k))/(np.sqrt(y_k(z,k+1))+np.sqrt(y_k(z,k)))**2+\
        (y_k(z,k)-y_k(z,k-1))/(np.sqrt(y_k(z,k))+np.sqrt(y_k(z,k-1)))/dist_k(z,k)\
         -dist_k(z,k)/2/np.sqrt(y_k(z,k))/(np.sqrt(y_k(z,k))+np.sqrt(y_k(z,k-1)))**2 for k in range(1,N)],
        [0] # we set it to zero since y_N is known and fixed.

    ])
    
    res=optimize.minimize(f,x0=make_guess(a,b,N),jac=gradf,bounds=bounds,constraints=dx_con,tol=1e-10,method='SLSQP',options={'ftol':1e-10,'maxiter': 1000})
    x=np.concatenate([[0],np.cumsum(res.x[:N])])
    y=-res.x[N:]
    return x,y

def travel_times(x,y):
    """Compute the travel times along each segment."""
    y=-y
    N=x.size-1
    return np.array([np.sqrt((x[k]-x[k-1])**2+(y[k]-y[k-1])**2)/(np.sqrt(y[k])+np.sqrt(y[k-1])) for k in range(1,N+1)])