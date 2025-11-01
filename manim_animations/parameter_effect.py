from manim import *
import numpy as np

class ParameterEffect(Scene):
    def construct(self):
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[-10, 10, 1],
            x_length=10,
            y_length=5,
            axis_config={
                "color": WHITE,
                "include_ticks": False,
            }
        ).to_edge(DOWN)

        x_label = Text("t").next_to(axes.x_axis, RIGHT)
        y_label = Text("h(t)").next_to(axes.y_axis, UP)
        labels = VGroup(x_label, y_label)


        def h(t, alpha, beta, gamma):
            return alpha * np.exp(t) * (1 - np.tanh(2*(t - beta))) * np.sin(gamma * t)

        alpha = ValueTracker(1)
        beta = ValueTracker(5)
        gamma = ValueTracker(5)

        graph = always_redraw(
            lambda: axes.plot(
                lambda t: h(t, alpha.get_value(), beta.get_value(), gamma.get_value()),
                x_range=[0, 10],
                color=ORANGE
            )
        )

        # title = always_redraw(
        #     lambda: Tex(
        #         f"$\\alpha={alpha.get_value():.2f}, \\; \\beta={beta.get_value():.2f}, \\; \\gamma={gamma.get_value():.2f}$",
        #         font_size=36
        #     ).to_edge(UP)
        # )
        title = always_redraw(
            lambda: Text(
                f"alpha={alpha.get_value():.2f}, beta={beta.get_value():.2f}, gamma={gamma.get_value():.2f}",
                font_size=32
            ).to_edge(UP)
        )

        self.play(Create(axes), Write(labels))
        self.play(Create(graph), Write(title))
        self.wait(1)

        # Animate alpha (amplitude)
        self.play(alpha.animate.set_value(2), run_time=3)
        self.play(alpha.animate.set_value(0.1), run_time=3)
        self.wait(1)

        # Animate beta (time shift)
        self.play(beta.animate.set_value(2), run_time=3)
        self.play(beta.animate.set_value(8), run_time=3)
        self.wait(1)

        # Animate gamma (frequency)
        self.play(gamma.animate.set_value(10), run_time=3)
        self.play(gamma.animate.set_value(2), run_time=3)
        self.wait(2)

