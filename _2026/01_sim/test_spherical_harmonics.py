from manimlib import *
import numpy as np
from scipy.special import lpmv
from sympy.physics.wigner import wigner_d
from sympy import N
from math import factorial

############################################################
# 1. 기본 Spherical Harmonics
############################################################

def Y(l, m, theta, phi):
    """
    Complex spherical harmonic without sph_harm
    """

    if m < 0:
        return (-1)**m * np.conj(Y(l, -m, theta, phi))

    norm = np.sqrt(
        ((2*l + 1)/(4*np.pi)) *
        factorial(l-m)/factorial(l+m)
    )

    P = lpmv(m, l, np.cos(theta))

    return norm * P * np.exp(1j*m*phi)


def Y_complex(l, m, theta, phi):
    """
    Complex spherical harmonic Y_l^m
    theta : [0, pi]
    phi   : [0, 2pi]
    """
    return Y(l, m, theta, phi)


def Y_real(l, m, theta, phi):
    """
    Real spherical harmonics
    """
    if m > 0:
        return np.sqrt(2) * np.real(Y_complex(l, m, theta, phi))
    elif m < 0:
        return np.sqrt(2) * np.imag(Y_complex(l, -m, theta, phi))
    else:
        return np.real(Y_complex(l, 0, theta, phi))


############################################################
# 2. 여러 모드 합성
############################################################

def combine_modes(modes, theta, phi, real=True):
    """
    modes: list of (l, m, coefficient)
    """
    result = 0

    for l, m, c in modes:
        if real:
            result += c * Y_real(l, m, theta, phi)
        else:
            result += c * Y_complex(l, m, theta, phi)

    return result

############################################################
# 3. Z축 회전
############################################################

def rotate_z_mode(l, m, alpha, theta, phi):
    """
    Z-axis rotation
    """
    return np.exp(1j * m * alpha) * Y_complex(l, m, theta, phi)


############################################################
# 4. 일반 Euler 회전 (α, β, γ)
############################################################

def rotate_general(l, modes, alpha, beta, gamma, theta, phi):
    """
    modes: list of (m, coefficient) for fixed l
    """
    result = 0

    for m, c in modes:
        rotated_component = 0

        for mp in range(-l, l + 1):
            D = complex(N(wigner_d(l, m, mp, beta))) \
                * np.exp(-1j * m * alpha) \
                * np.exp(-1j * mp * gamma)

            rotated_component += D * Y_complex(l, mp, theta, phi)

        result += c * rotated_component

    return result


############################################################
# 5. ManimGL Scene
############################################################

class SphericalHarmonicScene(ThreeDScene):

    def construct(self):

        axes = ThreeDAxes()
        self.add(axes)

        ####################################################
        # 설정할 모드
        ####################################################

        modes = [
            (3, 2, 1.0),
            (3, -1, 0.5),
            (4, 0, 0.4)
        ]

        ####################################################
        # Surface 정의
        ####################################################

        def surface_func(u, v):

            theta = u
            phi = v

            value = combine_modes(
                modes,
                theta,
                phi,
                real=True
            )

            r = 1 + 0.4 * value

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            return np.array([x, y, z])

        surface = ParametricSurface(
            surface_func,
            u_range=(0, PI),
            v_range=(0, TAU),
            resolution=(60, 60)
        )

        alpha_tracker = ValueTracker(0)

        def rotated_surface_func(u, v):

            theta = u
            phi = v

            value = 0

            for l, m, c in modes:
                value += c * np.real(
                    rotate_z_mode(l, m,
                                  alpha_tracker.get_value(),
                                  theta,
                                  phi)
                )

            r = 1 + 0.4 * value

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            return np.array([x, y, z])

        rotated_surface = always_redraw(
            lambda: ParametricSurface(
                rotated_surface_func,
                u_range=(0, PI),
                v_range=(0, TAU),
                resolution=(60, 60)
            )
        )

        self.play(ReplacementTransform(surface, rotated_surface))

        self.play(alpha_tracker.animate.set_value(TAU),
                  run_time=6,
                  rate_func=linear)

        self.wait()