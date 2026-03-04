from manimlib import *
import numpy as np
from scipy.special import lpmv
from sympy.physics.wigner import wigner_d
from sympy import N
from math import factorial

############################################################
# 1. Spherical Harmonics
############################################################

def Y(l, m, theta, phi):

    if m < 0:
        return (-1)**m * np.conj(Y(l, -m, theta, phi))

    norm = np.sqrt(
        ((2*l + 1)/(4*np.pi)) *
        factorial(l-m)/factorial(l+m)
    )

    P = lpmv(m, l, np.cos(theta))

    return norm * P * np.exp(1j*m*phi)


def Y_real(l, m, theta, phi):
    if m > 0:
        return np.sqrt(2) * np.real(Y(l, m, theta, phi))
    elif m < 0:
        return np.sqrt(2) * np.imag(Y(l, -m, theta, phi))
    else:
        return np.real(Y(l, 0, theta, phi))


def combine_modes(modes, theta, phi):
    result = 0
    for l, m, c in modes:
        result += c * Y_real(l, m, theta, phi)
    return result


############################################################
# 2. Scene
############################################################

class SphericalHarmonicScene(ThreeDScene):

    def construct(self):

        ####################################################
        # 모드 설정
        ####################################################

        modes = [
            (3, 2, 1.0),
            (3, -1, 0.5),
            (4, 0, 0.4)
        ]

        ####################################################
        # 3D Surface
        ####################################################

        def surface_func(u, v):

            value = combine_modes(modes, u, v)
            r = 1 + 0.4 * value

            x = r * np.sin(u) * np.cos(v)
            y = r * np.sin(u) * np.sin(v)
            z = r * np.cos(u)

            return np.array([x, y, z])

        surface = ParametricSurface(
            surface_func,
            u_range=(0, PI),
            v_range=(0, TAU),
            resolution=(50, 50)
        )

        self.play(ShowCreation(surface))
        self.wait(1)

        ####################################################
        # 2D 그래프 영역
        ####################################################

        axes = Axes(
            x_range=(0, PI, PI/4),
            y_range=(-1.5, 1.5, 0.5),
            width=8,
            height=4,
        ).to_edge(DOWN)

        self.play(FadeIn(axes))

        phi_fixed = 0
        graphs = []
        colors = [RED, GREEN, BLUE]

        ####################################################
        # 개별 모드 그래프
        ####################################################

        for i, (l, m, c) in enumerate(modes):

            graph = axes.get_graph(
                lambda theta, l=l, m=m, c=c:
                    c * Y_real(l, m, theta, phi_fixed),
                x_range=(0, PI),
                color=colors[i]
            )

            graphs.append(graph)

        ####################################################
        # 합성 그래프
        ####################################################

        total_graph = axes.get_graph(
            lambda theta:
                combine_modes(modes, theta, phi_fixed),
            x_range=(0, PI),
            color=YELLOW
        )

        ####################################################
        # 애니메이션
        ####################################################

        for g in graphs:
            self.play(ShowCreation(g), run_time=1)

        self.play(ShowCreation(total_graph), run_time=2)

        self.wait()