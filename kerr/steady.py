"""
Wigner function of a Kerr resonator
"""
from scipy.special import hyp2f1 as hypergeometric
from scipy.special import factorial
import numpy as np
import decimal


class Kerr(object):
    """
    A class to calculate necessary function for simulating a Kerr resonator.
    """
    def __init__(self, f, g, c):
        """
        All the system parameters go here
        
        Parameters
        ----------
        f, g, c: complex
            Dimensionless quantities formed using the system parameters

        m: int
            The index upto convergence calculated by the normalization function.
        """
        self.f = f
        self.g = g
        self.c = c

        self.m = None
        self.N = None

    def fm(self, m):
        """
        The function given by 

        $$ F_m(f, g, c) = (i\sqrt{g})^m \bigg[. _2F_1(-m, -c -i\frac{f}{\sqrt{g}};-2c;2)\bigg]$$

        Parameters
        ----------
        m: int
            The series index for the particular function

        Returns
        -------
        func: complex
            The value of the function used in the normalization composed of a hypergeometric function
        """
        f, g, c = self.f, self.g, self.c
        
        coeff = np.power(1j * np.sqrt(g), m)
        
        t2 = - c -(1j*f)/(np.sqrt(g))
        
        print(-m, t2, -2*c, 2)
        hyp = hypergeometric(-m, t2, -2*c, 2)

        return (coeff * hyp)


    def normalization(self, tol=0.000001, max_iter= 1000000):
        """
        The normalization constant calculated using the formula

        $$N = \sum_{m=0}^\inf \frac{2^m}{m!} |F_m(f, g,c)|^2$$

        The value of m is calculated until convergence and stored.

        Parameters
        ----------
        f, g, c: complex
            Dimensionless quantities formed using the system parameters

        tol: float
            The error tolerance to stop the iterations

        max_iter: int
            The maximum number of iterations to run for convergence.
        Returns
        -------
        N: float
            The normalization constant calculated till convergence or upto max_iter
        """
        f, g, c = self.f, self.g, self.c
        coefficient = lambda x: np.power(2, x)/factorial(x)

        m = 0
        N_old = coefficient(m) * np.absolute(self.fm(m))

        m += 1
        N_new = coefficient(m) * np.absolute(self.fm(m))

        while(np.absolute(N_new - N_old) > tol):
            N_old = N_new
            m += 1

            N_new += coefficient(m) * np.absolute(self.fm(m))

            if m > max_iter:
                break

        return N_new, m


    def correlation(i, j):
        """
        The correlation function defined using the normalization and fm

        Parameters
        ----------
        f, g, c: complex
            Dimensionless quantities formed using the system parameters

        m: int
            The truncation of the index upto which the factor should be calculated

        i, j: int
            The index (i, j) of the matrix to calculate the correlation function

        Returns
        -------
        corr: float
            The value of the correlation.
        """
        f, g, c = self.f, self.g, self.c

        if self.N is None:
            self.N, self.m = self.normalization()
            N = self.N
            m = self.m

        else:
            N = self.N
            m = self.m

        corr = 0

        coefficient = lambda x: np.power(2, x)/factorial(x)
        hyp = lambda x: self.fm(x + j)*(self.fm(x +i).conjugate())

        generator = (coefficient(idx) * hyp(idx) for idx in range(m))

        for term in generator:
            corr += term
        
        return corr/N

    def wigner(z):
        """
        The Wigner function

        Parameters
        ----------
        f, g, c: complex
            Dimensionless quantities formed using the system parameters

        m: int
            The truncation of the index upto which the factor should be calculated

        z: complex
            The value of z

        Returns
        -------
        W: float
            The real valued Wigner function.
        """        
        f, g, c = self.f, self.g, self.c

        if self.N is None:
            self.N, self.m = self.normalization()
            N = self.N
            m = self.m

        else:
            N = self.N
            m = self.m

        _wigner = 0

        coefficient = lambda x: np.pow(2*z.conjugate(), m)/factorial(x)
        generator = (coefficient(idx) * self.fm(idx) for idx in range(m))

        for term in generator:
            _wigner += term

        _wigner = np.power(np.absolute(_wigner), 2)

        wigner = (2./(N * np.pi)) * _wigner * np.exp(-2*np.power(abs(z),2))

        return wigner
